# Push Guide for Anisoap-torch

## What goes in the repo

Your `Anisoap-torch` repo at `github.com/Tejas7007/Anisoap-torch` should contain:

```
Anisoap-torch/
├── README.md                          # ← from this package
├── profile_baseline.py                # ← from this package
├── validate_final.py                  # ← from this package
├── anisoap/
│   └── representations/
│       ├── radial_basis.py            # ← from your AniSOAP fork (modified)
│       └── ellipsoidal_density_projection.py  # ← from your AniSOAP fork (modified)
├── rust/
│   ├── lib.rs                         # ← from this package (or your AniSOAP fork)
│   ├── ellip_expansion/
│   │   ├── mod.rs                     # ← from your AniSOAP fork
│   │   └── compute_moments.rs         # ← from your AniSOAP fork (modified)
│   └── Cargo.toml                     # ← from your AniSOAP fork (add rayon dep)
└── notebooks/
    ├── benzenes.xyz                   # ← test data (optional, for reproducibility)
    └── ellipsoids.xyz                 # ← test data (optional, for reproducibility)
```

## Terminal commands

```bash
# 1. Navigate to your Anisoap-torch repo (or clone if empty)
cd ~/research
git clone https://github.com/Tejas7007/Anisoap-torch.git
cd Anisoap-torch

# 2. Copy README and scripts from this package
# (download them from claude.ai outputs, or copy from ~/Downloads)
cp ~/Downloads/README.md .
cp ~/Downloads/profile_baseline.py .
cp ~/Downloads/validate_final.py .

# 3. Copy the MODIFIED source files from your AniSOAP fork
# These are the files you edited during our sessions
mkdir -p anisoap/representations
cp ~/AniSOAP/anisoap/representations/radial_basis.py anisoap/representations/
cp ~/AniSOAP/anisoap/representations/ellipsoidal_density_projection.py anisoap/representations/

# 4. Copy the Rust source files
mkdir -p rust/ellip_expansion
cp ~/AniSOAP/rust/lib.rs rust/
cp ~/AniSOAP/rust/ellip_expansion/compute_moments.rs rust/ellip_expansion/
cp ~/AniSOAP/rust/ellip_expansion/mod.rs rust/ellip_expansion/
cp ~/AniSOAP/Cargo.toml rust/

# 5. Optionally copy test data for reproducibility
mkdir -p notebooks
cp ~/AniSOAP/notebooks/benzenes.xyz notebooks/
cp ~/AniSOAP/notebooks/ellipsoids.xyz notebooks/

# 6. Stage and commit
git add -A
git status  # review what's being added

git commit -m "feat: full PyTorch rewrite achieving 3.8x speedup on benzenes, 15.7x on ellipsoids

- Eliminated per-pair Python loop (87k iterations → batched ops)
- Batched gaussian parameters (87k calls → 1 call)
- Batched Rust FFI (87k calls → 4 calls)
- Converted pipeline to PyTorch (einsum, solve, eigh, scatter_add)
- Added device parameter for GPU readiness
- All outputs validated identical to original"

git push origin main
```

## Note on file locations

The modified source files live in your AniSOAP fork at `~/AniSOAP/`.
Only the *modified* files go in `Anisoap-torch` — this repo is NOT a full
copy of AniSOAP, just the optimized components + benchmarks + documentation.
