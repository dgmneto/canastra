//! GA training for Canastra policy networks — pure Rust, no Python.
//!
//! This crate replaces the Python `canastra_train` package. The engine and
//! encoder crates are reused directly (no PyO3 boundary). The GPU forward
//! pass uses `tch-rs` (libtorch bindings), the same CUDA backend as PyTorch.
//!
//! ## Architecture
//!
//! - `seedstream` — deterministic seed streams (SplitMix64, same as Python)
//! - `elo` — ELO rating tracker
//! - `genome` — flat genome ↔ weights JSON, arch definitions
//! - `policy` — batched forward pass via tch-rs (WeightStack, logits_stacked)
//! - `ga` — GA core (elitism, tournaments, mutation, HOF, checkpoints)
//! - `pool` — batched game stepping (owns N engines, encode/apply per ply)
//! - `league` — self-play pairings, picker, drive loop
//! - `evaluate` — duplicate-deal paired evaluation
//! - `train` — CLI driver

pub mod elo;
pub mod evaluate;
pub mod ga;
pub mod genome;
pub mod league;
pub mod policy;
pub mod pool;
pub mod seedstream;

pub use genome::Arch;
