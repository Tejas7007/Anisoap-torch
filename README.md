# Anisoap-torch

Full PyTorch rewrite of [AniSOAP](https://github.com/cersonsky-lab/AniSOAP)'s numerical pipeline with batched operations — achieving **3.8× speedup on benzenes** and **15.7× on ellipsoids** on CPU, with GPU readiness built in.

This builds on the earlier [einsum-only optimization work](https://github.com/Tejas7007/cersonskylab-anisoap-optimization) and goes significantly further by eliminating all per-pair Python loops, batching every operation, and converting the full pipeline to PyTorch.

---

## Why This Exists

The original AniSOAP library computes anisotropic SOAP descriptors using NumPy. Our [previous benchmarking work](https://github.com/Tejas7007/cersonskylab-anisoap-optimization) identified `numpy.einsum` as 66–77% of runtime and swapped it to `torch.einsum`, yielding a **15–25% speedup**.

This repo goes deeper. By profiling with `cProfile` at the function level, we discovered the **real bottleneck** was not einsum itself (only 11% of actual compute) — it was the **Python for-loop** iterating over 87,174 atom pairs, calling functions individually for each pair. The previous profiling attributed loop overhead to einsum because einsum was the most expensive call *inside* each iteration.

---

## Results: Previous vs Current Approach

### Previous Approach: einsum-only swap
> *Swap `np.einsum` → `torch.einsum`, keep everything else NumPy*

| System | NumPy Baseline | PyTorch einsum | Speedup |
|---|---|---|---|
| Benzenes (50 frames) | 203.18s | 172.81s | **15% faster** |
| Ellipsoids (50 frames) | 1.56s | 1.17s | **25% faster** |

*CHTC Linux Cluster, x86_64, 1 CPU/job*

### Current Approach: Full PyTorch rewrite + batching
> *Eliminate Python loops, batch all operations, convert full pipeline to PyTorch*

| System | NumPy Baseline | Full PyTorch Rewrite | Speedup |
|---|---|---|---|
| Benzenes (50 frames) | 10.43s | 2.75s | **3.8× (280% faster)** |
| Ellipsoids (50 frames) | 0.408s | 0.026s | **15.7×** |

*Apple M4, macOS, single-threaded*

> **Note:** Wall times differ between platforms (CHTC cluster vs Mac M2), so absolute times aren't directly comparable. The key metric is the **relative speedup**, which shows the full rewrite delivers an order-of-magnitude greater improvement than the einsum-only swap.

### Why 3.8× vs 15%?

The einsum-only approach left the massive per-pair Python loop untouched. That loop was responsible for 85% of runtime:

| What we eliminated | Before | After | Improvement |
|---|---|---|---|
| Python loop overhead | 8.83s (85%) | 0.005s | **1,766×** |
| Per-pair gaussian params (87k calls) | 1.90s | 0.015s (1 call) | **127×** |
| Per-pair linalg.solve (174k calls) | 1.10s | 0.01s (8 calls) | **110×** |
| Per-pair einsum (610k calls) | 1.13s | 0.18s (64 calls) | **6.3×** |
| Per-pair Rust FFI (87k calls) | 3.02s | 2.11s (4 calls) | **1.4×** |
| Per-sample contraction | 0.32s | 0.18s | **1.8×** |

---

## Optimization Progression

| Stage | Benzenes Time | Cumulative Speedup |
|---|---|---|
| Original NumPy baseline | 10.43s | 1.0× |
| + Batched gaussian parameters | 8.53s | 1.2× |
| + Eliminated Python per-pair loop | 4.40s | 2.4× |
| + Batched Rust FFI (87k → 4 calls) | 3.44s | 3.0× |
| + Vectorized contract_pairwise_feat | 3.40s | 3.1× |
| + Full PyTorch conversion | **2.75s** | **3.8×** |

---

## Current Bottleneck

```
compute_moments_batch (Rust):  2.11s  (71% of total)
torch.einsum:                  0.18s  ( 6%)
contract_pairwise_feat:        0.18s  ( 6%)
metatensor overhead:           0.27s  ( 9%)
everything else:               0.23s  ( 8%)
```

The Rust `compute_moments` is now the hard floor — pure recurrence-relation math for trivariate Gaussian moments. Each pair costs ~24μs and Rayon parallelism showed no benefit at this granularity.

---

## What Was Changed

### 1. Batched Operations (Algorithmic)

**Before:** 87,174 individual Python function calls per neighbor pair
**After:** 4 batched calls (one per species-pair block)

```python
# BEFORE: per-pair loop (87k iterations)
for isample in nl_block.samples:
    r_ij = nl_block.values[isample]
    precision, center, constant = compute_gaussian_parameters(r_ij, lengths, rot)
    moments = compute_moments(precision, center, maxdeg)
    for l in range(lmax+1):
        np.einsum("mnpqr, pqr->mn", sph_to_cart[l], moments)

# AFTER: batched operations (4 calls total)
all_r_ij = nl_block.values[:, :, 0]                          # (N, 3)
all_precision, all_center, all_constant = batch_gaussian(...)  # 1 call
all_moments = compute_moments_batch(...)                       # 1 Rust FFI call
for l in range(lmax+1):
    torch.einsum("mnpqr, bpqr->bmn", sph_to_cart[l], all_moments_l)
```

### 2. Rust FFI Optimization

Added `compute_moments_batch_rust()` to process all pairs in a single Python→Rust crossing with GIL release:

```rust
pub fn compute_moments_batch_rust(
    dil_mats: &[f64],   // flattened (N, 3, 3)
    gau_cens: &[f64],   // flattened (N, 3)
    max_deg: i32,
    n_pairs: usize,
) -> PyResult<Vec<f64>>
```

### 3. Full PyTorch Backend

| Operation | NumPy | PyTorch |
|---|---|---|
| Linear solve | `np.linalg.solve` | `torch.linalg.solve` |
| Eigendecomposition | `np.linalg.eigh` | `torch.linalg.eigh` |
| Tensor contraction | `np.einsum` | `torch.einsum` |
| Scatter-add | `np.add.at` | `torch.scatter_add_` |
| Matrix sqrt inverse | SciPy-based | `torch.linalg.eigh` |

### 4. GPU Readiness

```python
# CPU (default)
edp = EllipsoidalDensityProjection(**hypers)

# GPU (future — when metatensor supports torch tensors)
edp = EllipsoidalDensityProjection(**hypers, device="cuda")
```

The `device` parameter propagates through all computation. Current limitation: metatensor `TensorBlock.values` are NumPy arrays, requiring CPU↔GPU transfers.

---

## Validation

All outputs verified identical to original:

- No NaNs, no Infs across all blocks
- Batched gaussian parameters match per-pair originals (100 random pairs, machine precision)
- Batched Rust moments match per-pair originals (50 random pairs, max diff = 0.00e+00)
- Final descriptor values match original pipeline exactly

---

## Files Modified (from [cersonsky-lab/AniSOAP](https://github.com/cersonsky-lab/AniSOAP))

| File | Changes |
|---|---|
| `anisoap/representations/radial_basis.py` | Full PyTorch rewrite; `torch.linalg.solve/eigh/einsum`, `compute_gaussian_parameters_batch()`, `device` param |
| `anisoap/representations/ellipsoidal_density_projection.py` | Full PyTorch rewrite; batched pairwise expansion, `torch.einsum`, `scatter_add_`, `device` param |
| `rust/ellip_expansion/compute_moments.rs` | Added `compute_moments_batch_rust()` with Rayon parallelism |
| `rust/lib.rs` | Exposed batched Rust function with GIL release via `py.allow_threads()` |
| `Cargo.toml` | Added `rayon = "1.10"` dependency |

---

## Profiling

### Hyperparameters
```
max_angular=6, radial_basis_name="gto", cutoff_radius=7.0, radial_gaussian_width=1.5
```

### Test Data
- `benzenes.xyz`: 50 frames, 2 species (H, C), high neighbor density → 87,174 pairs
- `ellipsoids.xyz`: 50 frames, 1 species, simple ellipsoidal particles

### Profiling Commands
```bash
# Benchmark
python profile_baseline.py

# Validation
python validate_final.py
```

---

## Future Work

1. **Port `compute_moments` to PyTorch** — Eliminate the 2.1s Rust bottleneck by implementing the recurrence relations as a PyTorch kernel (enables GPU execution)
2. **metatensor-torch integration** — Use `metatensor.torch.TensorBlock` to avoid numpy↔torch conversion overhead
3. **`torch.compile()`** — JIT compilation of the einsum + scaling pipeline
4. **Larger benchmarks** — Profile on realistic MD trajectories (1000+ frames)

---

## Related Repositories

- [cersonsky-lab/AniSOAP](https://github.com/cersonsky-lab/AniSOAP) — Original library
- [Tejas7007/AniSOAP](https://github.com/Tejas7007/AniSOAP) — Fork with modifications
- [Tejas7007/cersonskylab-anisoap-optimization](https://github.com/Tejas7007/cersonskylab-anisoap-optimization) — Previous einsum-only benchmarking

## License

MIT
