//! Deterministic seed streams — SplitMix64 finalizer.
//!
//! Same mixer family the engine uses for hand seeds. Everything derives from
//! (run_seed, generation), so a resumed run regenerates exactly the seeds it
//! had without storing RNG state.

const MASK: u64 = u64::MAX;

pub fn splitmix64(x: u64) -> u64 {
    let mut x = (x.wrapping_add(0x9E37_79B9_7F4A_7C15)) & MASK;
    x = ((x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9)) & MASK;
    x = ((x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB)) & MASK;
    (x ^ (x >> 31)) & MASK
}

/// `count` distinct u64 seeds shared by every pairing of one generation.
pub fn generation_seeds(run_seed: u64, generation: u32, count: usize) -> Vec<u64> {
    let base = splitmix64(
        (run_seed & MASK) ^ ((generation as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)) & MASK,
    );
    (0..count)
        .map(|i| splitmix64(base.wrapping_add(i as u64)))
        .collect()
}
