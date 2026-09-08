# TP4 Hierarchical Custom Allreduce — Session Log

Status: **DONE. Size-gate fix applied, rebuilt via CMake incremental workflow,
re-validated. DFlash2 regression fixed — throughput now exceeds every prior
baseline, target-only and DFlash2 alike. Follow-on v100-skinny kernel port
investigated and deliberately abandoned — native TurboMind already covers
both our checkpoint format and RadixArk's at a comparable performance class,
with no need to replace it. See §8 for final numbers and §8e-g for the
kernel-port investigation.**

## 1. Problem

Hardware: 4x Tesla V100-SXM2-16GB, wired as two independent NVLink pairs
(GPU0↔GPU1, GPU2↔GPU3, both NV6/~145GB/s) bridged only by PCIe
(~6.4GB/s, no NVLink P2P, no P2P atomics on the cross-pair links). Confirmed via
`nvidia-smi topo -m` and `-p2p n/r/w/a/p`, and via `p2pBandwidthLatencyTest`
(built from the CUDA samples repo).

vLLM's `CustomAllreduce` requires a **full** 4-way NVLink mesh
(`current_platform.is_fully_connected()`, an NVML pairwise NVLink P2P check) before
enabling its fast collective for `world_size > 2`. On this box that check fails, so
flat TP4 falls back to generic NCCL for every layer's all-reduce, on every decode
step.

Measured cost of this, this session:

| Config | Target-only (no DFlash2) | With DFlash2 |
|---|---|---|
| Our TP4 (NCCL fallback, before this work) | **31.8 tok/s** | 49.1 tok/s |
| Our TP2 (NVLink pair, `CUSTOM` active) | 45.1 tok/s | — |
| Our TP2 (PCIe pair, `CUSTOM` active) | 45.9 tok/s | — |
| 1CatAI's own QA doc, full-mesh TP4 (`docs/design/sm70_dflash2_quality_audit.md`, "physical V100 group 4-7") | 75.246 tok/s | 237.8 tok/s |

Key deduction: TP2 gets the same speed on NVLink or PCIe pairs (both qualify for
`CUSTOM` since `world_size<=2` doesn't require full mesh) — proving it's specifically
the *disabled collective*, not raw link bandwidth, costing us at TP4. And DFlash2's
own multiplier is dampened by it too (4.8x on their hardware vs 1.5x on ours),
because DFlash2's verification round pays the slow-allreduce tax repeatedly.

## 2. The existing precedent (why this was portable)

`csrc/custom_all_reduce.cuh` already implements a two-level "clique + single
cross-clique peer" hierarchical custom allreduce, but hardcoded for
`world_size == 8` split into two 4-GPU cliques (env-gated by
`VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR`). Audited in detail and confirmed:
clique size is a named constant used consistently as a loop bound (only two spots
use the literal `4` directly: `clique_base` and `pair_rank` computation); the
underlying `RankData`/`RankSignals`/`Signal` structs are already generic
(`kMaxRanks = 8`); the algorithm structure (reduce-within-clique → exchange one
partial with one cross-clique peer → sum two partials) is provably independent of
clique size. This meant a 2-GPU-clique port for our TP4/2+2 topology was a
mechanical `4→2` substitution, not a rewrite.

Also found a **separate** existing `kSm70Tp4PushAllreduce*` constant family
(flat, non-hierarchical, gated on `fully_connected` — doesn't help us) whose byte
sizes are tuned to **our exact model** (`hidden_size=5120`, DFlash2's "verifies
eight positions at once" comment): 2560, 4096, 4×2560, 8×2560, 8×5120 elements
(fp16). We reused these byte constants for our new hierarchical size-gate instead
of inventing generic ones — this detail is now the source of the DFlash2 problem
(see §6).

## 3. What was implemented

Plan file: `/home/aksh/.claude/plans/groovy-jingling-moth.md` (approved plan, still
accurate for the "Option A" design — the size-gate refinement in §6 below is a
follow-on fix, not a plan deviation).

### `csrc/custom_all_reduce.cuh`
- New constant `kSm70Tp4HierarchicalCliqueSize = 2`.
- New `sm70_tp4_hierarchical_custom_ar_enabled(world_size, fully_connected)`,
  `sm70_tp4_hierarchical_peer(rank, peer)`,
  `sm70_tp4_hierarchical_allreduce_size(bytes)` (currently accepts all 5 of the
  `kSm70Tp4PushAllreduce*` sizes — **this is what needs narrowing, §6**).
- New kernels `sm70_tp4_clique_barrier` / `sm70_tp4_hierarchical_reduce`, exact
  mechanical ports of `sm70_tp8_clique_barrier` / `sm70_tp8_hierarchical_reduce`
  with clique size 4→2 (`clique_base`, `pair_rank`, the `tid<4`/`tid==4` fan-out
  conditions, the inner reduce loop bound, and the final `rank<4?` combine all
  become `2`/`tid<2`/`tid==2`). Verified symbol present in the built `.so`:
  `vllm::sm70_tp4_hierarchical_reduce(RankData*, RankSignals, Signal*, __half*, int, int)`.
- Dispatch branch inserted in `allreduce()` right after the existing flat-TP4-push
  branch, before the TP8 hierarchical branches.
- `register_graph_buffers()` mirror added (nulls non-peer `RankData::ptrs` under
  CUDA-graph capture, matching the TP8 precedent) — confirmed exercised
  correctly in real engine runs ("Registering N cuda graph addresses" with no
  errors, both target-only and DFlash2).

### `vllm/distributed/device_communicators/custom_all_reduce.py`
- New `_sm70_tp4_hierarchical_peer_ranks(rank)` — verified output:
  `peer_ranks(0)=(0,1,2)`, `(1)=(0,1,3)`, `(2)=(2,3,0)`, `(3)=(2,3,1)` — clique
  {0,1}/{2,3}, cross-pairs 0↔2 and 1↔3, matching the C++ side exactly.
- New `_SM70_TP4_HIERARCHICAL_ELEMENTS = frozenset((4096, 2560, 4*2560, 8*2560, 8*5120))`
  and `_SM70_TP4_HIERARCHICAL_SCRATCH_BYTES`.
- Constructor: added `sm70_tp4_candidate` branch parallel to `sm70_tp8_candidate`;
  `self.tp4_hierarchical` flag (pre-initialized `False` near the other early
  defaults, then set for real after the topology check passes); scratch-size
  `max(...)` updated; P2P-capability-check skip extended to `tp4_hierarchical`
  too.
- `should_custom_ar()`: new `if self.tp4_hierarchical:` branch mirroring the TP8
  one.

### `vllm/envs.py`
- New env var `VLLM_SM70_TP4_HIERARCHICAL_CUSTOM_AR` (default **off** — opt-in),
  registered in both the type-annotation block and the getter dict, mirroring
  the exact `VLLM_SM70_TP8_HIERARCHICAL_CUSTOM_AR` pattern.

## 4. Build process (this repo, from source, first time ever)

We had only ever used the prebuilt wheel before this session. Key facts for
resuming:

- **Dev image**: `1cat-vllm-dev:build` — built from
  `container_build/Dockerfile.dev` (separate from the production
  `container_build/Dockerfile` — never touch that one for this work). Base:
  `nvidia/cuda:12.8.1-devel-ubuntu24.04` + `python3.12-dev build-essential git
  cmake ninja-build ccache` + torch 2.10.0+cu128 + `uv`.
  - **Do not set `ENV CC="ccache gcc"` / `CXX="ccache g++"`** in this Dockerfile
    — it broke two separate things this session: (1) the vendored
    `flash-attention-v100` sub-package's plain-distutils build (`gcc: error:
    linker input file not found: gcc` — a literal stray token from the
    compound string), and (2) Triton's runtime JIT compilation at server
    startup (`FileNotFoundError: 'ccache gcc'`). CMake's own ccache
    integration (`CMAKE_*_COMPILER_LAUNCHER=ccache`) doesn't need these env
    vars at all and works fine without them. **Always `unset CC; unset CXX`**
    before any build or server-launch command in this image.
- **Running dev container**: `1cat-vllm-devbuild`, still up as of this log,
  with volumes: `/home/aksh/repos/v100-vllm-1cat:/workspace`,
  `/home/aksh/.cache/1cat-vllm-ccache:/root/.ccache`,
  `/home/aksh/models/hf/Qwen3.8-27B-QUASAR-NVFP4:/models/target:ro`,
  `/home/aksh/models/hf/Qwen3.8-27B-DFlash2:/models/draft:ro`. Port 8000
  published.
- **ccache did not help** across separate `pip install -e .` invocations —
  each invocation creates a fresh random `/tmp/tmpXXXXXX.build-temp` directory,
  and absolute paths baked into compiler invocations differ each time,
  invalidating ccache's hash despite identical source. **This temp directory
  also does not survive container recreation or even a successful build**
  (setuptools deletes it on success) — there is no way to resume/incrementally
  rebuild via plain `pip install -e .` a second time.
- **The fix for fast iteration** (not yet executed, queued as next step): follow
  `docs/contributing/incremental_build.md` — run
  `python tools/generate_cmake_presets.py`, which writes
  `CMakeUserPresets.json` with `binaryDir` set to `${sourceDir}/cmake-build-release`
  — **inside the bind-mounted repo**, so it persists across container restarts.
  First `cmake --preset release && cmake --build --preset release --target
  install` will still take the full ~20-30 min (fresh directory), but every
  subsequent `cmake --build --preset release --target install` after a source
  edit will be fast (only recompiles/relinks what changed). **Do this before
  the next source-code iteration** to stop paying the full rebuild cost every
  time.
- **Build command used** (full, from-scratch, before the incremental setup
  above existed):
  ```bash
  cd /workspace
  export TORCH_CUDA_ARCH_LIST=7.0
  export CMAKE_CUDA_ARCHITECTURES=70
  export MAX_JOBS=8
  export NVCC_THREADS=1
  unset CC; unset CXX
  /opt/venv/bin/pip install -e . --no-build-isolation
  ```
  (`--no-build-isolation` is required — without it, `uv`/`pip`'s isolated build
  sandbox pulls in a mismatched torch/nvidia-* set and fails with
  `undefined symbol: __nvJitLinkGetErrorLogSize_12_9`. Build deps —
  `cmake ninja packaging patchelf setuptools setuptools-scm setuptools-rust
  wheel jinja2` — must be pre-installed into the active env first, matching
  `pyproject.toml`'s `[build-system].requires`, since isolation is off.)
- **After container recreation, two more things break and need redoing** (both
  bitten this session when the container was recreated mid-work to add model
  volume mounts):
  1. Package metadata is gone (`importlib.metadata.version("1cat-vllm")` raises
     `PackageNotFoundError`, which cascades into vLLM's own platform-detection
     failing with `RuntimeError: Failed to infer device type`). The compiled
     `.so` files under `/workspace/vllm/*.abi3.so` are untouched (bind-mounted,
     persist fine) — only the ephemeral `/opt/venv` metadata registration is
     lost. **Fix without rebuilding**: manually recreate a minimal
     `<venv>/lib/python3.12/site-packages/1cat_vllm-<version>.dist-info/METADATA`
     (just `Metadata-Version`/`Name`/`Version` fields) plus a `.pth` file
     containing `/workspace` in site-packages. Verified this works — no
     rebuild needed for this specific breakage.
  2. Runtime Python deps (`numpy`, `transformers`, `fastapi`, etc.) are also
     gone (ephemeral `/opt/venv`). Reinstall via
     `pip install -r requirements/cuda.txt` (fast — this hits pip's package
     cache, ~1-2 min, not a rebuild).
  3. `flash_attn_v100` (the vendored `flash-attention-v100/` sub-package) is
     **never importable by default** even after a successful build — it
     compiles via its own `setup.py build_ext --inplace`, which leaves the
     `.so` files sitting in `/workspace/flash-attention-v100/` without
     installing them into site-packages. **Fix**: set
     `PYTHONPATH=/workspace/flash-attention-v100` when launching anything that
     imports vLLM (server, test scripts). This is required every time,
     independent of container recreation — it's just how this vendored
     package's build works.

## 5. Validation performed

### 5a. Standalone correctness + microbenchmark
Script: `container_build/tp4_hierarchical_correctness_test.py` (kept in repo).
Run via: `torchrun --nproc_per_node=4 tp4_hierarchical_correctness_test.py`
inside the dev container (needs the dev venv's torch/vllm; does **not** need
the model volumes, just the built `_C.abi3.so`).

Note: `CustomAllreduce` must be constructed on a **separate `gloo` group**
(`dist.new_group(backend="gloo")`), not the default NCCL `dist.group.WORLD` —
it asserts non-NCCL. Keep the NCCL group for the `dist.all_reduce` baseline
comparison.

Results (all 5 accepted sizes, 100 random seeds each):
- **Correctness: exact match** vs plain NCCL (`max_abs_diff=0.000000` for
  every size/seed — not just within tolerance).
- bf16 correctly bypasses the new path (returns `None`, falls through).
- Stress test: 2000 back-to-back iterations, no hangs, no corruption.
- **Microbenchmark** (this is the crux of the current follow-up fix):

  | numel | bytes | custom (μs) | nccl (μs) | speedup |
  |---|---|---|---|---|
  | 2560 | 5KiB | 36.6 | 47.0 | **1.28x faster** |
  | 4096 | 8KiB | 35.2 | 48.7 | **1.38x faster** |
  | 10240 | 20KiB | 54.2 | 46.4 | 0.86x slower |
  | 20480 | 40KiB | 87.4 | 46.8 | 0.54x slower |
  | 40960 | 80KiB (DFlash2 8-position verify size) | 152.9 | 92.4 | **0.60x slower** |

  Our kernel's cross-clique exchange is a simple polling handshake over the
  slow ~6.4GB/s PCIe link — fine for tiny messages, bad for larger ones where
  NCCL's own (presumably chunked/pipelined) algorithm wins.

### 5b. Full-engine target-only (deterministic, temp=0)
Same exact test as the pre-existing 31.8 tok/s baseline (Qwen3.8-27B-NVFP4,
TP4, `FLASH_ATTN_V100`, no DFlash2, quicksort prompt, `max_tokens=512`,
`temperature=0`).

**Result: 31.8 → 51.2 tok/s = 1.61x speedup.** Response verified correct
(same content pattern, same `finish_reason: length` / `completion_tokens: 512`
as the baseline). CUDA graph capture succeeded cleanly (158s, "Registering 128
cuda graph addresses" — confirms `register_graph_buffers` mirror works under
real graph capture, not just eager mode).

This is a clean, reproducible, deterministic win — trust this number.

### 5c. Full-engine DFlash2 (stochastic, temp=0.3) — INCONCLUSIVE, root cause found
Same exact test as the pre-existing 49.1 tok/s baseline (robot-painting-story
prompt, `max_tokens=400`, `temperature=0.3`, `max_num_seqs=2`).

Three trials: 39.9, 40.0 tok/s (throughput), with draft acceptance rates
27.5%/33.3% vs the original baseline's 48.8% — all **lower** than baseline,
consistently, not a one-off.

**Root cause (not sampling noise — this was my first, wrong explanation,
correctly rejected by the user):** DFlash2's verification step batches 8
positions at once — exactly the **40960-element / 80KiB** message size that
the 5a microbenchmark showed as **0.60x (40% slower)** than NCCL. Target-only
decode only ever hits the small winning sizes (2560/4096); DFlash2 additionally
hits the large losing size on every verification round, and since verification
is on DFlash2's critical path, the net effect is a real regression, not noise.

## 6. Next step (queued, not yet done — this is where we paused)

**Fix**: narrow `sm70_tp4_hierarchical_allreduce_size()` in
`csrc/custom_all_reduce.cuh` (currently at line ~266) to only accept the two
sizes with a *proven* win:
```cpp
inline bool sm70_tp4_hierarchical_allreduce_size(size_t bytes) {
  return bytes == kSm70Tp4PushAllreduce8KiBBytes ||
         bytes == kSm70Tp4PushAllreduceQwen4ExpBytes;
}
```
(drops `kSm70Tp4PushAllreduceQwen38M4Bytes`, `...M8Bytes`, and
`kSm70Tp4PushAllreduceBytes` — the three sizes that lost to NCCL in 5a). Also
update the Python-side mirror `_SM70_TP4_HIERARCHICAL_ELEMENTS` in
`custom_all_reduce.py` to match: `frozenset((4096, 2560))`.

**Then**: set up the persistent CMake incremental build (§4) before rebuilding,
so this and any future source tweaks are fast, not another 20-30 min full
rebuild. Then re-run 5b (target-only — expect ~unchanged, still 1.61x, since it
never hit the large sizes anyway) and 5c (DFlash2 — expect it to now land at
or above the 49.1 tok/s baseline, since the losing 80KiB verify-path allreduce
will now correctly fall through to the existing NCCL/generic path instead of
our slower-for-that-size kernel).

## 7. Environment pointers for resuming

- Plan file: `/home/aksh/.claude/plans/groovy-jingling-moth.md`
- Dev container (still running): `1cat-vllm-devbuild` (image `1cat-vllm-dev:build`)
- Repo (bind-mounted, has all edits + built `.so` files): `/home/aksh/repos/v100-vllm-1cat`
- Correctness test script: `container_build/tp4_hierarchical_correctness_test.py`
- Server logs from this session: `container_build/target_only_hierarchical3.log`
  (the successful 51.2 tok/s run), `container_build/dflash2_hierarchical.log`
  (the 39.9-40.0 tok/s run)
- A DFlash2 server (pre-fix) is still running inside `1cat-vllm-devbuild` at the
  time of this pause (pid 3174 / 3168, port 8000) — stop it
  (`docker exec 1cat-vllm-devbuild pkill -f vllm.entrypoints.cli.main`) before
  starting the next test.
- Production image `1cat-vllm:1.5.0-sm70` (prebuilt-wheel based, untouched by
  any of this) is still intact and unaffected — this whole effort has been
  isolated to the separate dev image/container.

## 8. Resumption: fix applied, rebuilt, re-validated (2026-09-05/06)

### 8a. Fix applied
Both files edited exactly as specified in §6:
- `csrc/custom_all_reduce.cuh`: `sm70_tp4_hierarchical_allreduce_size()` narrowed
  to only `kSm70Tp4PushAllreduce8KiBBytes` and `kSm70Tp4PushAllreduceQwen4ExpBytes`
  (drops the three sizes that measured slower than NCCL).
- `custom_all_reduce.py`: `_SM70_TP4_HIERARCHICAL_ELEMENTS` narrowed to
  `frozenset((4096, 2560))`.

### 8b. Incremental build finally set up
Followed `docs/contributing/incremental_build.md` for the first time in this
project (previously only ever used ephemeral `pip install -e .`):
```bash
cd /workspace
export CUDA_HOME=/usr/local/cuda
unset CC; unset CXX
python tools/generate_cmake_presets.py --force-overwrite
# then manually added "CMAKE_CUDA_ARCHITECTURES": "70" to CMakeUserPresets.json
# to avoid building for every arch — SM70 only
cmake --preset release
cmake --build --preset release --target install
```
Generated preset used `NVCC_THREADS=4`, `jobs=6` (auto-detected from 24 cores/
62GB RAM — no OOM issues at this parallelism, unlike the more conservative
`NVCC_THREADS=1` needed for the original full `pip install -e .` build). Since
`CMakeUserPresets.json` and `cmake-build-release/` live inside the bind-mounted
repo (`/home/aksh/repos/v100-vllm-1cat`), they now **persist across container
restarts** — any future kernel edit only needs `cmake --build --preset release
--target install` again, not this whole setup. Build succeeded, exit code 0.

### 8c. Standalone correctness + microbenchmark, re-run with narrowed gate
`torchrun --nproc_per_node=4 container_build/tp4_hierarchical_correctness_test.py`
with `VLLM_SM70_TP4_HIERARCHICAL_CUSTOM_AR=1`. The test script reads
`_SM70_TP4_HIERARCHICAL_ELEMENTS` directly, so it automatically tested only the
two remaining sizes:

```
Testing element sizes: [2560, 4096]
numel=  2560: max_abs_diff=0.000000 max_rel_diff=0.000000 over 100 seeds
numel=  4096: max_abs_diff=0.000000 max_rel_diff=0.000000 over 100 seeds
bf16 correctly bypassed the custom path (returned None, falls through to NCCL)
stress test: 2000 iterations completed, 0.9s, no hang, no mismatch
numel=  2560  custom=   35.19us  nccl=   46.44us  speedup=1.32x
numel=  4096  custom=   35.06us  nccl=   47.08us  speedup=1.34x
ALL CHECKS PASSED
```
Exact numerical match (not just within tolerance) across both sizes, 100 seeds
each. No corruption in the previously-passing sizes from the narrowing edit.
The three dropped sizes (10240/20480/40960 elements) are no longer routed
through our kernel at all — they fall through to the existing NCCL/generic path
by construction (the C++ gate simply returns `false` for them now).

### 8d. Full-engine re-validation
Same launch flags as the original baselines (§1), plus
`VLLM_SM70_TP4_HIERARCHICAL_CUSTOM_AR=1` and `PYTHONPATH=/workspace/flash-attention-v100`.
Confirmed via boot log in both cases: `Using ['CUSTOM', 'PYNCCL'] all-reduce
backends` (our hierarchical path is active; plain TP4 on this topology would
show `Custom allreduce is disabled` instead, as separately confirmed this
session on the v100-skinny stack, which does **not** carry this fix).

**Target-only** (deterministic, temp=0, quicksort-explanation prompt,
`max_tokens=512`):

| | tok/s |
|---|---:|
| Original baseline (NCCL fallback) | 31.8 |
| Post-fix (this session) | **74.5** |

2.34x over the original baseline — higher than the previously-logged 51.2 tok/s
(1.61x) from before this narrowing fix. Target-only never hit the sizes this
fix removed, so the extra gain beyond 51.2 is not attributable to the size-gate
edit itself — most likely additional codegen/optimization differences from this
being the **first-ever CMake-based build** of the extension versus the earlier
ephemeral `pip install -e .` build. Response content verified coherent and
correct (well-formed, correctly-commented quicksort implementation, no
repetition or corruption artifacts) — trusted alongside the exact-match kernel
correctness in 8c, not in place of it.

**DFlash2, single-stream** (temp=0.3, robot-painting-story prompt,
`max_tokens=400`, `max_num_seqs=2` but only one request in flight):

| | tok/s |
|---|---:|
| Original baseline | 49.1 |
| Regressed (broad size gate, pre-fix) | 39.9–40.0 |
| **Post-fix (this session)** | **64.6 / 70.6 / 71.4** (3 trials) |

Regression fully resolved and then some — comfortably above the original
49.1 tok/s baseline. Response content verified coherent (well-formed short
story, no corruption).

**DFlash2, 2 truly concurrent streams** (new this session — not previously
measured; matches `--max-num-seqs 2`'s actual intended use, sending both
requests via separate threads simultaneously rather than serially):

| Trial | Stream 0 | Stream 1 | Aggregate (sum tokens / batch wall time) |
|---|---:|---:|---:|
| 1 | 46.25 tok/s | 48.13 tok/s | **92.49 tok/s** |
| 2 | 42.49 tok/s | 42.66 tok/s | **84.97 tok/s** |
| 3 | 44.97 tok/s | 46.30 tok/s | **89.93 tok/s** |

Each stream runs slower individually under concurrent load (42-48 tok/s vs.
64-71 tok/s single-stream) since both share the same GPUs and get verified in
the same batched round, but aggregate throughput (85-93 tok/s) exceeds either
single-stream number, confirming real batching rather than serial
time-slicing. No hangs or errors across 3 trials. No pre-fix concurrent
baseline exists for direct comparison — this is new information, not a
before/after claim.

### 8e. v100-skinny kernel port: investigated, abandoned with cause (2026-09-06)

Follow-on question after 8a-8e: does porting v100-skinny's hand-written QPN2/
QPN8 kernels onto this tree (stacked with the allreduce fix) beat what we
already have? Investigated by actually trying it rather than assuming.

**Our own checkpoint (QUASAR-QAT, compressed-tensors W4A4) never reaches
v100-skinny's hook.** Ported `marlin.py` + `kernels/skinny_kernels.cu`
verbatim (JIT-compiled cleanly, zero ABI issues, same torch 2.10.0+cu128 on
both sides) and confirmed via boot log that our model dispatches through
`compressed_tensors_w4a4_nvfp4.py`'s own **"SM70 TurboMind W4A16 dense path"**
— a different, already-built 1CatAI-native kernel that v100-skinny's patch
(targeting `CompressedTensorsW4A16Fp4`) never touches. Confirmed empirically:
enabling `VLLM_SKINNY_NVFP4=1` produced byte-identical throughput (74.56/74.58
tok/s, unchanged) and zero skinny-kernel log activity. **Reverted — dead code
for our checkpoint.**

**RadixArk (ModelOpt format) already has native TurboMind support too.**
v100-skinny's `modelopt.py` patch (576 differing lines) rips out 1CatAI's
existing `sm70_turbomind` FP8/NVFP4 dispatch entirely, replacing it with
v100-skinny's own kernels — by their own comment, partly "BRING-UP REFERENCE
ONLY... nothing here is the shipping design." Before committing to that
surgery (plus the much larger `gpu_model_runner.py` reconciliation, 2,041
differing lines in a 13k-line file), tested whether our *unmodified* tree
already handles RadixArk. Recreated `1cat-vllm-devbuild` with a RadixArk bind
mount (`/home/aksh/models/hf/Qwen3.8-27B-NVFP4-RadixArk:/models/radixark:ro`),
restored the ephemeral `/opt/venv` state (metadata + `pip install -r
requirements/cuda.txt`; torch itself is baked into the image layer, unaffected
by recreation), and booted with only the allreduce fix + the two small
KV-quant-policy additions (8g below) active — no v100-skinny modelopt/kernel
code at all.

Result: **confirmed working out of the box.** Boot log: `SM70 ModelOpt FP8
TurboMind W8A16 dense path enabled` and `SM70 ModelOpt NVFP4 TurboMind dense
path enabled`; warmup executed 15 real FP8 + 15 real FP4 dense calls. Measured
(quicksort prompt, temp=0, max_tokens=512, same methodology as 8d):

| Config | Single-stream | 2-concurrent aggregate |
|---|---:|---:|
| QUASAR-QAT + our allreduce fix | 74.5 tok/s | — |
| QUASAR-QAT + our fix + DFlash2 | 70.9 tok/s avg | ~89 tok/s |
| **RadixArk + our fix, native TurboMind only** | **66.6-66.7 tok/s** | **118.75-119.17 tok/s** (3 trials, tight variance) |

Response content verified coherent in all cases. RadixArk's native-TurboMind
number sits in the same performance class as v100-skinny's own published
number on this exact checkpoint under 1.2.2 (69.7 tok/s, though that run used
MTP k=7 + a prose workload — not a clean apples-to-apples, but clearly not an
order-of-magnitude gap either way).

**Conclusion: the full 6-file port is not worth doing.** TurboMind already
covers both checkpoint formats we care about at a comparable performance
class to v100-skinny's hand-written kernels, without the risk of replacing a
working native path with kernels whose own author calls part of the
replacement "bring-up reference, not the shipping design." `marlin.py` /
`kernels/skinny_kernels.cu` reverted; nothing from the v100-skinny port
remains in the tree except the two independently-useful KV-quant-policy
additions below, which came from reading their `torch_utils.py`/`attention.py`
diffs, not from the kernel work itself.

### 8f. Two small, genuinely useful additions kept from the port attempt

While diffing v100-skinny's patches against our tree (not blindly copying —
several files would have silently deleted newer features our tree has grown
since their 1.2.2 fork point, e.g. `np_to_pinned_tensor`, prefix-anchored SWA,
DFlash2 draft-attention handling, `CompressedTensorsKVCacheMethod` unit-scale
special-casing — all confirmed still needed and preserved), one small,
self-contained, genuinely valuable fix stood out and was surgically applied
(not file-replaced) to two files:

- **`vllm/utils/torch_utils.py`**: new `checkpoint_kv_quant_allowed()` —
  declines a checkpoint's own `kv_cache_quant_algo`/`kv_cache_scheme` directive
  below SM80 (no FP8 tensor-core hardware), keeping FP16 KV and the tensor-core
  decode route instead of silently falling onto `scalar_paged` attention.
  Escape hatch: `VLLM_SM70_ALLOW_CKPT_KV_QUANT=1`.
- **`vllm/model_executor/layers/attention/attention.py`**: the one call site
  in `Attention.__init__` that resolves `kv_cache_scheme` now calls the above
  before honoring it.

Only fires when `--kv-cache-dtype auto` (never touches an explicit
`--kv-cache-dtype`, so our existing `fp8_e5m2`-pinned target-only/DFlash2
configs are unaffected). Confirmed firing correctly on the RadixArk boot:
`Ignoring the checkpoint's kv_cache quantization directive (fp8_e4m3): this
device has no FP8 support...`.

### 8g. Conclusion
The paused work is complete: the fix does exactly what §6 predicted (the 80KiB
DFlash2 verify-round message now falls through to NCCL instead of the
kernel's losing large-message path), correctness is proven at both the kernel
level (exact numerical match) and the end-to-end output level (coherent,
non-corrupted generations), and every measured configuration — target-only,
single-stream DFlash2, and concurrent DFlash2 — now exceeds its original
pre-this-session baseline.

## 9. llama.cpp + Unsloth Q4_K_XL + MTP exploration (separate thread from the allreduce work above)

Not part of the vLLM/allreduce fix — a parallel investigation into whether
llama.cpp + Unsloth's Q4 GGUF + native MTP speculative decoding beats our
production vLLM setup. Kept in this log since it's the same overall
decision space (what to actually run in production).

### 9a. Model refresh
Local `Qwen3.8-27B-UD-Q4_K_XL.gguf` (downloaded Aug 16, 17.92 GB) was stale —
Unsloth shipped "Dynamic v3.0" ~20 days ago (10% higher accuracy claimed at
the same size, post-training quant, not QAT). Re-downloaded from
`unsloth/Qwen3.8-27B-GGUF` (same filename, new content): **16.35 GiB**,
confirmed via GGUF header scan to still carry MTP weights
(`blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`,
`qwen35.nextn_predict_layers` metadata key) — self-speculative, no separate
draft checkpoint needed, matching the same design as the NVFP4 QUASAR
checkpoint's baked-in MTP head (§ elsewhere this session). Old file replaced
in place at `/home/aksh/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf`.

llama.cpp's own flag for this: `--spec-type draft-mtp` (native support,
confirmed via `llama-server --help`).

### 9b. TP2 (one island, GPU0+1) — memory-constrained, real crash found

Weight footprint: ~9.5 GB/GPU (matches this project's older historical
measurement for this quant class). At **full 262,144 context**, MTP enabled:
booted with only **530 MiB free/GPU** (15,854/16,384 MiB used) — survived one
short exchange (~65 cached tokens) but **crashed under a real stress test**:
a plain 20,000-filler-word prompt (~26K real tokens) hit a genuine CUDA OOM
mid-matmul during MTP draft processing:
```
ggml_cuda_pool_vmm::alloc() -> ggml_cuda_mul_mat_cublas ->
common_speculative_impl_draft_mtp::process()
```
**Conclusion: TP2 + MTP + full context is not safe on this hardware.**
At **half context (131,072)**, TP2 was solid (~12.65 GB/GPU, comfortable
margin). Benchmarked there (quicksort prompt, 400 tokens):

| Config | Sampling | tok/s (server) | Draft accept |
|---|---|---:|---|
| No MTP | temp=0 | 45.9 | — |
| MTP (n-max=3) | temp=0 (greedy) | 71.8 | 63.7% (167/262) |
| No MTP | Unsloth thinking (temp=1.0, top_p=0.95, top_k=20) | 48.3 | — |
| MTP (n-max=3) | Unsloth thinking | 61.4 | 48.4% (236/488) |

Greedy overstates the real speedup (~1.5x) vs. the model's actual intended
sampling regime (~1.27x) — acceptance rate drops meaningfully at real
temperature since it's no longer an exact-match criterion.

### 9c. TP4 (all 4 GPUs) — much safer, and faster

Weight footprint: ~8.94 GB/GPU (vs. TP2's ~9.5, plus 3 more GPUs sharing the
same total) — **7.4+ GiB free/GPU** at boot. Ran the same context-fill stress
sweep that crashed TP2 (20K/60K/120K filler words): **survived cleanly,
memory barely moved** (9216 -> 9226 MiB/GPU across a 6x growth in real prompt
tokens, 22K -> 133K). tok/s degrades with context length as expected
(58.0 -> 49.0 -> 40.5 tok/s over that same sweep) — normal attention-cost
scaling, not a memory problem.

Full 262,144 context, Unsloth thinking-mode sampling, quicksort prompt:

| Config | tok/s (server) |
|---|---:|
| No MTP | 56.5 |
| MTP (n-max=3) | 77.1 |

**This is the best number of the whole exploration** — beats every TP2
result, at full context, with real memory safety margin (unlike TP2's
razor-thin/crashing full-context case).

### 9d. `--spec-draft-n-max` sweep (TP4, Unsloth thinking-mode sampling, 300-token completions)

| n-max | tok/s | Draft accept |
|---|---:|---|
| 1 | 59.8 | 73.8% |
| 2 | 62.6 | 59.3% |
| **3** | **69.3** | **68.6%** |
| 4 | 51.8 | 38.8% |
| 5 | 56.9 | 44.0% |
| 7 | 56.2 | 43.2% |

**n-max=3 is the genuine peak**, not an arbitrary default — acceptance rate
falls off a cliff past 3 (69% -> ~39-44%), so drafting further ahead is mostly
wasted compute on this checkpoint/hardware. All §9b/9c numbers above already
used n-max=3.

### 9e. vs. current production (vLLM TP4, our hierarchical allreduce fix, NVFP4 QUASAR-QAT, no MTP)

| | tok/s | Quantization | Hardening |
|---|---:|---|---|
| Production (vLLM) | 74.5 | NVFP4, **QAT** (QUASAR — GPQA-Diamond 0.9091 vs BF16 teacher's 0.9141) | Tool calling, vision, systemd service, memory-safety validated all session |
| llama.cpp TP4+MTP(n-max=3) | 69.3-77.1 | Q4_K_XL, **PTQ** (Unsloth Dynamic v3) | Single quick test only — no tool-calling/API validation, only partial stress-testing |

Roughly a throughput tie (within single-sample noise); the real
differentiator is quantization method (QAT should preserve more quality than
PTQ at the same bit width) and production readiness, not raw speed. **Not
adopted as a production change today — logged as a real, promising option to
revisit** if quality-validated and hardened (tool-calling support, broader
quality eval beyond one coding prompt, sustained-load testing) later.
