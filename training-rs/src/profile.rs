//! Coarse per-generation profiling accumulators (Phase 0).
//!
//! Atomic counters accumulated across worker + server threads. Printed once
//! at generation end via [`report`]. All times are nanoseconds. This is
//! throwaway instrumentation — Phase 1 rewrites the rollout without it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// Server (GPU) thread — one thread, serial processing.
pub static SRV_IDLE_NS: AtomicU64 = AtomicU64::new(0); // rx.recv() blocking
pub static SRV_H2D_NS: AtomicU64 = AtomicU64::new(0); // Tensor::from_vec uploads
pub static SRV_GPU_NS: AtomicU64 = AtomicU64::new(0); // forward_picks (kernel + D2H + argmax)
pub static SRV_REQS: AtomicU64 = AtomicU64::new(0); // forward requests served
pub static SRV_ROWS: AtomicU64 = AtomicU64::new(0); // rows forwarded total

// Worker threads — summed across all workers (overlap with server).
pub static WKR_ENCODE_NS: AtomicU64 = AtomicU64::new(0); // pool.encode()
pub static WKR_APPLY_NS: AtomicU64 = AtomicU64::new(0); // pool.apply()
pub static WKR_FWD_NS: AtomicU64 = AtomicU64::new(0); // gpu.forward() wall (mutex+channel+gpu)
pub static WKR_PLIES: AtomicU64 = AtomicU64::new(0); // ply iterations

pub static GEN_WALL_NS: AtomicU64 = AtomicU64::new(0); // whole-generation wall

pub fn reset() {
    SRV_IDLE_NS.store(0, Ordering::Relaxed);
    SRV_H2D_NS.store(0, Ordering::Relaxed);
    SRV_GPU_NS.store(0, Ordering::Relaxed);
    SRV_REQS.store(0, Ordering::Relaxed);
    SRV_ROWS.store(0, Ordering::Relaxed);
    WKR_ENCODE_NS.store(0, Ordering::Relaxed);
    WKR_APPLY_NS.store(0, Ordering::Relaxed);
    WKR_FWD_NS.store(0, Ordering::Relaxed);
    WKR_PLIES.store(0, Ordering::Relaxed);
    GEN_WALL_NS.store(0, Ordering::Relaxed);
}

/// A span accumulator: records elapsed ns into `target` on drop.
pub struct Span<'a> {
    start: Instant,
    target: &'a AtomicU64,
}

impl<'a> Span<'a> {
    pub fn new(target: &'a AtomicU64) -> Self {
        Self {
            start: Instant::now(),
            target,
        }
    }

    pub fn add(target: &'a AtomicU64, d: Duration) {
        target.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    }
}

impl Drop for Span<'_> {
    fn drop(&mut self) {
        let ns = self.start.elapsed().as_nanos() as u64;
        self.target.fetch_add(ns, Ordering::Relaxed);
    }
}

pub fn record(target: &AtomicU64, ns: u64) {
    target.fetch_add(ns, Ordering::Relaxed);
}

/// Print the breakdown table for one generation.
pub fn report(games: usize) {
    let wall = GEN_WALL_NS.load(Ordering::Relaxed);
    let srv_idle = SRV_IDLE_NS.load(Ordering::Relaxed);
    let srv_h2d = SRV_H2D_NS.load(Ordering::Relaxed);
    let srv_gpu = SRV_GPU_NS.load(Ordering::Relaxed);
    let srv_reqs = SRV_REQS.load(Ordering::Relaxed);
    let srv_rows = SRV_ROWS.load(Ordering::Relaxed);
    let wkr_enc = WKR_ENCODE_NS.load(Ordering::Relaxed);
    let wkr_app = WKR_APPLY_NS.load(Ordering::Relaxed);
    let wkr_fwd = WKR_FWD_NS.load(Ordering::Relaxed);
    let wkr_plies = WKR_PLIES.load(Ordering::Relaxed);

    let srv_busy = srv_h2d + srv_gpu;
    let avg_rows = if srv_reqs > 0 {
        srv_rows as f64 / srv_reqs as f64
    } else {
        0.0
    };
    let pct = |v: u64| (v as f64 / wall as f64) * 100.0;
    let sec = |v: u64| v as f64 / 1e9;

    eprintln!("┌─ profile: generation breakdown ─────────────────────────────");
    eprintln!("│ wall          {:>10.3}s  (games={})", sec(wall), games);
    eprintln!(
        "│ games/s       {:>10.1}",
        games as f64 / sec(wall).max(1e-9)
    );
    eprintln!("│ ── server (GPU) thread ──");
    eprintln!(
        "│   busy        {:>10.3}s  {:>5.1}%  (H2D {:.3}s + GPU {:.3}s)",
        sec(srv_busy),
        pct(srv_busy),
        sec(srv_h2d),
        sec(srv_gpu)
    );
    eprintln!(
        "│   idle        {:>10.3}s  {:>5.1}%  (starving: workers not feeding)",
        sec(srv_idle),
        pct(srv_idle)
    );
    eprintln!(
        "│   requests    {:>10}    avg rows/req {:>6.1}    plies {}",
        srv_reqs, avg_rows, wkr_plies
    );
    eprintln!("│   rows total  {:>10}    plies {}", srv_rows, wkr_plies);
    eprintln!("│ ── workers (summed, overlap with server) ──");
    eprintln!(
        "│   encode      {:>10.3}s  {:>5.1}%",
        sec(wkr_enc),
        pct(wkr_enc)
    );
    eprintln!(
        "│   apply       {:>10.3}s  {:>5.1}%",
        sec(wkr_app),
        pct(wkr_app)
    );
    eprintln!(
        "│   fwd-wait    {:>10.3}s  {:>5.1}%  (mutex+channel+gpu, blocked on server)",
        sec(wkr_fwd),
        pct(wkr_fwd)
    );
    eprintln!("└──────────────────────────────────────────────────────────────");
}
