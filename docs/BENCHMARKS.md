# Benchmark Methodology and Technical Notes

## Scope

The measurements compare the pinned upstream AniSOAP numerical path with the batched PyTorch and Rust implementation in this repository. Each comparison used the same workload and hardware within the experiment.

## Recorded Apple M4 results

| Workload | Frames | Upstream | Optimized | Speedup | Runtime reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Benzenes | 50 | 10.43 s | 2.75 s | 3.79× | 73.6% |
| Ellipsoids | 50 | 0.408 s | 0.026 s | 15.69× | 93.6% |

The source values are stored in [`benchmarks/results.json`](../benchmarks/results.json).

## Workloads

### Benzenes

- 50 trajectory frames
- hydrogen and carbon species
- 87,174 pair samples in the recorded run
- `max_angular=6`
- `radial_basis_name="gto"`
- `cutoff_radius=7.0`
- `radial_gaussian_width=1.5`

### Ellipsoids

- 50 trajectory frames
- one particle species
- the same descriptor hyperparameters

Both datasets are loaded from the pinned upstream AniSOAP revision during bootstrap.

## What changed

The optimization targets execution granularity across the full pairwise path:

1. neighbor metadata is gathered for a complete species-pair block;
2. PyTorch computes Gaussian precision matrices, transformed centers, and constants in batch;
3. the Python layer calls the Rust moment recurrence once per species-pair block;
4. PyTorch contracts moment tensors with spherical-to-Cartesian transforms;
5. `scatter_add_` aggregates pair values into center-level descriptor blocks;
6. the inherited radial-basis path orthonormalizes the result.

The modified files are listed in `NOTICE`. They remain structurally close to upstream AniSOAP to keep the patch reviewable and suitable for possible upstreaming.

## Interpreting the numbers

Speedup is calculated as:

```text
upstream runtime / optimized runtime
```

Runtime reduction is calculated as:

```text
1 - optimized runtime / upstream runtime
```

For example, a 3.79× speedup means the optimized run used about 26.4% of the upstream wall-clock time. It does not mean a 279% runtime reduction.

## Reproduction

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
python scripts/benchmark.py --frames 50 --repeats 3
```

Add `--profile` to write `cProfile` output. A strict upstream comparison should use a second environment containing the unmodified pinned revision rather than importing two versions of `anisoap` into one process.

## Numerical and device boundaries

- PyTorch performs batched linear algebra and contractions.
- The Rust extension currently accepts NumPy-backed CPU arrays.
- metatensor blocks are constructed from NumPy arrays.
- Non-CPU execution therefore introduces synchronization and transfer boundaries.

The implementation is described as device-aware, not end-to-end GPU-native.

## Caveats

- Wall-clock measurements depend on processor, BLAS, compiler, dependency versions, thermal state, thread settings, and background load.
- The original benchmark environment did not preserve every thread-control variable.
- The two workloads do not cover the complete AniSOAP parameter space.
- Repeated measurements should include a warm-up and report both minimum and mean runtime.
