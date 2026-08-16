//! Pinned-memory H2D microbenchmark.
//!
//! Measures raw host→device bandwidth for pageable vs pinned memory, on the
//! default stream. Separates "the PCIe link is slow" from "our transfers are
//! slow". Run with:
//!
//! ```bash
//! target\release\canastra-h2d-bench.exe
//! ```

use std::time::Instant;

fn main() {
    let size_mb = 256;
    let iters = 100;
    let n = (size_mb * 1_000_000) / 4; // f32 elements

    let ctx = std::sync::Arc::new(cudarc::driver::CudaContext::new(0).expect("CUDA context"));
    let stream = ctx.default_stream();

    println!("has_async_alloc: {}", ctx.has_async_alloc());

    // ── Pageable ──
    let pageable: Vec<f32> = vec![1.0; n];
    let _ = stream.clone_htod(&pageable).expect("warmup pageable");
    stream.synchronize().expect("sync");
    let began = Instant::now();
    for _ in 0..iters {
        let _dev = stream.clone_htod(&pageable).expect("pageable copy");
    }
    stream.synchronize().expect("sync");
    let pageable_elapsed = began.elapsed().as_secs_f64();
    let pageable_gbs = (size_mb as f64 * iters as f64) / pageable_elapsed / 1_000.0;

    // ── Pinned ──
    let pinned = unsafe { ctx.alloc_pinned::<f32>(n) }.expect("pinned alloc");
    let _ = stream.clone_htod(&pinned).expect("warmup pinned");
    stream.synchronize().expect("sync");
    let began = Instant::now();
    for _ in 0..iters {
        let _dev = stream.clone_htod(&pinned).expect("pinned copy");
    }
    stream.synchronize().expect("sync");
    let pinned_elapsed = began.elapsed().as_secs_f64();
    let pinned_gbs = (size_mb as f64 * iters as f64) / pinned_elapsed / 1_000.0;

    // ── Pre-allocated + memcpy (no per-iter alloc) ──
    let mut dev_pre = stream.alloc_zeros::<f32>(n).expect("pre-alloc");
    let _ = stream
        .memcpy_htod(&pinned, &mut dev_pre)
        .expect("warmup memcpy");
    stream.synchronize().expect("sync");
    let began = Instant::now();
    for _ in 0..iters {
        stream
            .memcpy_htod(&pinned, &mut dev_pre)
            .expect("pinned memcpy");
    }
    stream.synchronize().expect("sync");
    let prealloc_elapsed = began.elapsed().as_secs_f64();
    let prealloc_gbs = (size_mb as f64 * iters as f64) / prealloc_elapsed / 1_000.0;

    // ── Transfer stream (non-default) + pre-alloc ──
    let xfer_stream = ctx.new_stream().expect("transfer stream");
    let mut dev_xfer = stream.alloc_zeros::<f32>(n).expect("pre-alloc xfer");
    let _ = xfer_stream
        .memcpy_htod(&pinned, &mut dev_xfer)
        .expect("warmup xfer");
    xfer_stream.synchronize().expect("sync");
    let began = Instant::now();
    for _ in 0..iters {
        xfer_stream
            .memcpy_htod(&pinned, &mut dev_xfer)
            .expect("xfer memcpy");
    }
    xfer_stream.synchronize().expect("sync");
    let xfer_elapsed = began.elapsed().as_secs_f64();
    let xfer_gbs = (size_mb as f64 * iters as f64) / xfer_elapsed / 1_000.0;

    // ── Link state (after sustained load) ──
    let link = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "(nvidia-smi unavailable)".to_string());

    println!("=== H2D microbenchmark ===");
    println!("buffer: {} MB, {} iterations", size_mb, iters);
    println!(
        "pageable (alloc+copy):    {:.1}s = {:.1} GB/s",
        pageable_elapsed, pageable_gbs
    );
    println!(
        "pinned (alloc+copy):      {:.1}s = {:.1} GB/s",
        pinned_elapsed, pinned_gbs
    );
    println!(
        "pinned pre-alloc memcpy:  {:.1}s = {:.1} GB/s",
        prealloc_elapsed, prealloc_gbs
    );
    println!(
        "pinned xfer-stream memcpy:{:.1}s = {:.1} GB/s",
        xfer_elapsed, xfer_gbs
    );
    println!(
        "speedup pre-alloc vs alloc+copy: {:.1}x",
        prealloc_gbs / pageable_gbs
    );
    println!(
        "speedup xfer-stream vs default:  {:.1}x",
        xfer_gbs / prealloc_gbs
    );
    println!("link (after): {}", link);
}
