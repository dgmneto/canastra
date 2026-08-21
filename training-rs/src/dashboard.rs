//! In-process live dashboard for `canastra-train`, enabled with `--dashboard`.
//!
//! Spawns a tiny HTTP server on a background thread. The training loop pushes
//! each completed generation record straight into shared in-memory state; the
//! server renders the current snapshot into a single HTML page on every
//! request, and the page reloads itself every 15 seconds. No files are read —
//! the numbers shown are exactly what the trainer holds in memory at request
//! time. No WebSocket, no extra crates: just `std::net` + `serde_json`.
//!
//! # Intra-generation live view
//!
//! [`LiveStats`] carries the *within-generation* progress (games finished,
//! plies, per-individual running differentials) from the lockstep loop to the
//! page. It is engineered so it can never become a sync point for training:
//! per-ply updates are two atomic stores, and the heavier leaderboard
//! recompute is throttled to ~4×/s behind a `try_lock` — if the server is
//! reading, the update is simply skipped that ply.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::league::GameMeta;
use crate::pool::Pool;

/// The dashboard HTML template, with a placeholder the server replaces with
/// the live snapshot JSON before serving. Kept in a sibling file so the page is
/// easy to edit without touching Rust.
const DASHBOARD_HTML: &str = include_str!("dashboard.html");
/// Token inside `DASHBOARD_HTML` that the server swaps for the snapshot JSON.
const SNAPSHOT_TOKEN: &str = "/*SNAP*/null";
/// Token inside `DASHBOARD_HTML` that the server swaps for the refresh interval.
const REFRESH_TOKEN: &str = "__REFRESH_SECS__";

/// Refresh interval advertised to the browser (seconds).
const REFRESH_SECS: u32 = 15;

/// Minimum time between leaderboard recomputes on the training thread.
const LEADERBOARD_THROTTLE: Duration = Duration::from_millis(250);

/// Phase of the training loop inside the current generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// No generation running yet.
    Idle,
    /// Self-play league (the lockstep rollout).
    League,
    /// Anchored evaluation against the frozen opponents.
    Anchors,
    /// ES update / bookkeeping after the league.
    Update,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::League => "league",
            Phase::Anchors => "anchors",
            Phase::Update => "update",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => Phase::League,
            2 => Phase::Anchors,
            3 => Phase::Update,
            _ => Phase::Idle,
        }
    }
}

/// Running stats for one individual (roster index) in the live generation:
/// completed paired comparisons, the summed score differential, and wins
/// (win = 1, tie = 0.5) — the same arithmetic `fitness::score_generation`
/// applies, folded as games finish instead of at the end.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiveGenome {
    pub games: u32,
    pub diff_sum: f64,
    pub wins: f64,
}

#[derive(Default)]
struct LiveBook {
    leaderboard: Vec<LiveGenome>,
    last_finished: usize,
    last_scan: Option<Instant>,
}

/// Intra-generation live state, shared between the training thread and the
/// dashboard server. The hot path (once per ply) touches only atomics; the
/// leaderboard lives behind its own small mutex that the trainer acquires
/// with `try_lock` only, so server reads can never stall the lockstep loop.
pub struct LiveStats {
    gen: AtomicU32,
    phase: AtomicU8,
    /// f64 stored as bits (atomics carry no f64).
    sigma_bits: AtomicU64,
    roster: AtomicUsize,
    games_total: AtomicUsize,
    games_done: AtomicUsize,
    plies: AtomicU64,
    updated_bits: AtomicU64,
    book: Mutex<LiveBook>,
}

impl Default for LiveStats {
    fn default() -> Self {
        Self {
            gen: AtomicU32::new(0),
            phase: AtomicU8::new(0),
            sigma_bits: AtomicU64::new(0),
            roster: AtomicUsize::new(0),
            games_total: AtomicUsize::new(0),
            games_done: AtomicUsize::new(0),
            plies: AtomicU64::new(0),
            updated_bits: AtomicU64::new(0),
            book: Mutex::new(LiveBook::default()),
        }
    }
}

impl LiveStats {
    /// Reset for a new generation's league phase (training thread).
    fn reset(&self, gen: u32, sigma: f64, roster: usize) {
        self.gen.store(gen, Ordering::Relaxed);
        self.phase.store(Phase::League as u8, Ordering::Relaxed);
        self.sigma_bits.store(sigma.to_bits(), Ordering::Relaxed);
        self.roster.store(roster, Ordering::Relaxed);
        self.games_total.store(0, Ordering::Relaxed);
        self.games_done.store(0, Ordering::Relaxed);
        self.plies.store(0, Ordering::Relaxed);
        self.updated_bits.store(now_secs().to_bits(), Ordering::Relaxed);
        if let Ok(mut b) = self.book.try_lock() {
            *b = LiveBook::default();
        }
    }

    /// Render as JSON for the `/live` endpoint (server thread).
    fn live_json(&self) -> String {
        serde_json::to_string(&self.live_value()).unwrap_or_else(|_| "{}".to_string())
    }

    fn live_value(&self) -> Value {
        let leaderboard: Vec<Value> = {
            let book = self.book.lock().unwrap_or_else(|e| e.into_inner());
            book.leaderboard
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    serde_json::json!({
                        "i": i,
                        "g": g.games,
                        "ds": g.diff_sum,
                        "w": g.wins,
                    })
                })
                .collect()
        };
        serde_json::json!({
            "gen": self.gen.load(Ordering::Relaxed),
            "phase": Phase::from_u8(self.phase.load(Ordering::Relaxed)).as_str(),
            "sigma": f64::from_bits(self.sigma_bits.load(Ordering::Relaxed)),
            "roster": self.roster.load(Ordering::Relaxed),
            "games_total": self.games_total.load(Ordering::Relaxed),
            "games_done": self.games_done.load(Ordering::Relaxed),
            "plies": self.plies.load(Ordering::Relaxed),
            "updated_at": f64::from_bits(self.updated_bits.load(Ordering::Relaxed)),
            "now": now_secs(),
            "leaderboard": leaderboard,
        })
    }
}

/// Live training state, shared between the trainer thread and the dashboard
/// server thread.
pub struct DashboardState {
    pub config: Value,
    pub target_generations: u32,
    pub run_dir: String,
    pub started_at: f64,
    pub current_gen: u32,
    pub status: String,
    pub gen_started_at: f64,
    pub last_gen_wall_s: Option<f64>,
    pub generations: Vec<Value>,
    pub total_games: u64,
    pub total_wall_s: f64,
    pub updated_at: f64,
    /// Intra-generation live stats — shared handle, updated by the trainer
    /// without touching the outer state mutex.
    pub live: Arc<LiveStats>,
}

impl DashboardState {
    pub fn new(config: Value, target_generations: u32, current_gen: u32, run_dir: String) -> Self {
        let now = now_secs();
        Self {
            config,
            target_generations,
            run_dir,
            started_at: now,
            current_gen,
            status: "starting".to_string(),
            gen_started_at: now,
            last_gen_wall_s: None,
            generations: Vec::new(),
            total_games: 0,
            total_wall_s: 0.0,
            updated_at: now,
            live: Arc::new(LiveStats::default()),
        }
    }

    /// Render the whole state as a JSON string for the page to consume.
    pub fn snapshot(&self) -> String {
        let snap = serde_json::json!({
            "config": self.config,
            "target_generations": self.target_generations,
            "run_dir": self.run_dir,
            "started_at": self.started_at,
            "current_gen": self.current_gen,
            "status": self.status,
            "gen_started_at": self.gen_started_at,
            "last_gen_wall_s": self.last_gen_wall_s,
            "generations": self.generations,
            "total_games": self.total_games,
            "total_wall_s": self.total_wall_s,
            "updated_at": self.updated_at,
            "live": self.live.live_value(),
            "now": now_secs(),
        });
        serde_json::to_string(&snap).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Handle held by the trainer to push updates into the dashboard.
pub struct DashboardHandle {
    state: Arc<Mutex<DashboardState>>,
    live: Arc<LiveStats>,
}

impl DashboardHandle {
    pub fn new(state: Arc<Mutex<DashboardState>>) -> Self {
        let live = state
            .lock()
            .map(|s| s.live.clone())
            .unwrap_or_else(|_| Arc::new(LiveStats::default()));
        Self { state, live }
    }

    /// Mark the start of a generation (for the live "elapsed in gen" display).
    pub fn set_gen_start(&self, gen: u32) {
        if let Ok(mut s) = self.state.lock() {
            s.current_gen = gen;
            s.status = "running".to_string();
            s.gen_started_at = now_secs();
            s.updated_at = s.gen_started_at;
        }
    }

    /// Reset the intra-generation live view for a new league phase. `roster`
    /// is the population size plus HOF opponents actually scheduled to play.
    pub fn begin_league(&self, gen: u32, sigma: f64, roster: usize) {
        self.live.reset(gen, sigma, roster);
    }

    /// Announce which phase of the generation the trainer is in (shown as the
    /// live status in the UI).
    pub fn set_phase(&self, phase: Phase) {
        self.live.phase.store(phase as u8, Ordering::Relaxed);
        self.live
            .updated_bits
            .store(now_secs().to_bits(), Ordering::Relaxed);
    }

    /// Per-ply live update from the lockstep loop. Two atomic stores, plus a
    /// leaderboard recompute at most every [`LEADERBOARD_THROTTLE`] behind a
    /// `try_lock` — if the server holds the book, this ply's update is
    /// dropped and the next one catches up. Never blocks.
    pub fn observe(&self, pool: &Pool, meta: &[GameMeta], plies: u64) {
        let live = &self.live;
        live.plies.store(plies, Ordering::Relaxed);
        let done = pool.finished_count();
        live.games_done.store(done, Ordering::Relaxed);
        if live.games_total.load(Ordering::Relaxed) == 0 && !meta.is_empty() {
            live.games_total.store(meta.len(), Ordering::Relaxed);
        }

        let mut book = match live.book.try_lock() {
            Ok(b) => b,
            Err(_) => return, // server is reading — skip, never block
        };
        if book.last_finished == done {
            return;
        }
        if let Some(t) = book.last_scan {
            if t.elapsed() < LEADERBOARD_THROTTLE {
                return;
            }
        }

        // Recompute per-individual running stats from the games that have
        // finished so far. Games sit in `batch_layout` order, so adjacent
        // pairs (2k, 2k+1) are the two seatings of one deal — the same
        // duplicate-deal pairing `fitness::score_generation` uses, folded
        // early: a pair only counts once both seatings are done.
        let roster = live.roster.load(Ordering::Relaxed);
        let results = pool.snapshot_results();
        let mut acc = vec![LiveGenome::default(); roster];
        let mut g = 0;
        while g + 1 < results.len() {
            if let (Some((_, s0, _, _, _)), Some((_, s1, _, _, _))) = (results[g], results[g + 1])
            {
                let (a, b, _) = meta[g];
                if a < roster && b < roster {
                    let diff = ((s0[0] - s0[1]) + (s1[1] - s1[0])) as f64 / 2.0;
                    acc[a].games += 1;
                    acc[b].games += 1;
                    acc[a].diff_sum += diff;
                    acc[b].diff_sum -= diff;
                    let (wa, wb) = if diff > 0.0 {
                        (1.0, 0.0)
                    } else if diff < 0.0 {
                        (0.0, 1.0)
                    } else {
                        (0.5, 0.5)
                    };
                    acc[a].wins += wa;
                    acc[b].wins += wb;
                }
            }
            g += 2;
        }
        book.leaderboard = acc;
        book.last_finished = done;
        book.last_scan = Some(Instant::now());
        live.updated_bits.store(now_secs().to_bits(), Ordering::Relaxed);
    }

    /// Push a completed generation record (the same value written to
    /// `generations.jsonl`) straight into memory.
    pub fn record_generation(&self, record: &Value) {
        if let Ok(mut s) = self.state.lock() {
            if let Some(games) = record.get("games").and_then(Value::as_u64) {
                s.total_games += games;
            }
            if let Some(wall) = record.get("wall_s").and_then(Value::as_f64) {
                s.total_wall_s += wall;
                s.last_gen_wall_s = Some(wall);
            }
            if let Some(gen) = record.get("gen").and_then(Value::as_u64) {
                s.current_gen = gen as u32;
            }
            s.generations.push(record.clone());
            s.updated_at = now_secs();
        }
    }

    pub fn finish(&self) {
        if let Ok(mut s) = self.state.lock() {
            s.status = "done".to_string();
            s.updated_at = now_secs();
        }
    }
}

/// Spawn the dashboard HTTP server on a background thread.
pub fn spawn(state: Arc<Mutex<DashboardState>>, host: &str, port: u16) {
    let bind = format!("{host}:{port}");
    thread::spawn(move || {
        let listener = match TcpListener::bind(&bind) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("dashboard: bind {bind} failed: {e}");
                return;
            }
        };
        eprintln!("dashboard: http://{bind}  (auto-refresh {REFRESH_SECS}s)");
        for conn in listener.incoming() {
            let Ok(stream) = conn else { continue };
            let st = state.clone();
            thread::spawn(move || handle_conn(stream, st));
        }
    });
}

fn handle_conn(mut stream: TcpStream, state: Arc<Mutex<DashboardState>>) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    // Read the request line + headers (up to \r\n\r\n) so the socket is drained
    // before we respond. We only need the request target from the first line.
    let mut data = Vec::with_capacity(512);
    let mut rbuf = [0u8; 1024];
    loop {
        match std::io::Read::read(&mut stream, &mut rbuf) {
            Ok(0) => break,
            Ok(n) => {
                data.extend_from_slice(&rbuf[..n]);
                if data.ends_with(b"\r\n\r\n") || data.ends_with(b"\n\n") {
                    break;
                }
                if data.len() > 8192 {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    let path = request_path(&data);
    match path.as_str() {
        "/" | "/index.html" | "" => {
            let snap = state
                .lock()
                .map(|s| s.snapshot())
                .unwrap_or_else(|_| "{}".to_string());
            // Escape `<`, `>`, `&` so a `</script>` substring inside a JSON
            // string value can't prematurely close the <script> element.
            let snap = snap.replace('<', "\\u003c").replace('>', "\\u003e").replace('&', "\\u0026");
            let body = DASHBOARD_HTML
                .replacen(SNAPSHOT_TOKEN, &snap, 1)
                .replacen(REFRESH_TOKEN, &REFRESH_SECS.to_string(), 1);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
        // Small, fast JSON of just the intra-generation live state — polled
        // by the page every second between full reloads. The Arc is cloned
        // out under a brief lock so serialization never holds the trainer.
        "/live" => {
            let live = state
                .lock()
                .map(|s| s.live.clone())
                .unwrap_or_else(|_| Arc::new(LiveStats::default()));
            let body = live.live_json();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
        _ => {
            let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(resp.as_bytes());
        }
    }
}

/// Parse the request target from an HTTP request line buffer ("GET /path HTTP/1.1").
fn request_path(buf: &[u8]) -> String {
    let line_end = buf.iter().position(|&b| b == b'\r' || b == b'\n').unwrap_or(buf.len());
    let line = String::from_utf8_lossy(&buf[..line_end]);
    let mut parts = line.split_whitespace();
    let _method = parts.next();
    let target = parts.next().unwrap_or("/");
    target.split('?').next().unwrap_or("/").to_string()
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
