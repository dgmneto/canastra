//! GA training for Canastra policy networks — pure Rust, no Python.
//!
//! This crate replaces the Python `canastra_train` package. The engine and
//! encoder crates are reused directly (no PyO3 boundary). The forward pass
//! uses `candle-core` (pure Rust ML framework, CUDA via cudarc).
//!
//! ## Architecture
//!
//! - `seedstream` — deterministic seed streams (SplitMix64, same as Python)
//! - `elo` — ELO rating tracker
//! - `genome` — flat genome ↔ weights JSON, arch definitions
//! - `policy` — lockstep grouped-matmul forward via candle (WeightStack,
//!   forward_picks/forward_scores; bf16 on CUDA, fp32 on CPU; GPU argmax)
//! - `ga` — GA core (elitism, tournaments, mutation, HOF, checkpoints)
//! - `pool` — batched game stepping (owns N engines, encode/apply per ply)
//! - `league` — self-play pairings and the lockstep ply driver
//! - `evaluate` — duplicate-deal paired evaluation
//! - `bin/train` — CLI driver
//! - `bin/bench` — benchmark one generation

pub mod anchors;
pub mod elo;
pub mod es;
pub mod evaluate;
pub mod ga;
pub mod genome;
pub mod league;
pub mod policy;
pub mod pool;
pub mod seedstream;

#[cfg(feature = "profile")]
pub mod profile;

pub use genome::Arch;
