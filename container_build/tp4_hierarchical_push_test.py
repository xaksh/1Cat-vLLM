"""
Correctness + in-graph microbenchmark for the 2+2 TP4 hierarchical all-reduce
(pull kernel `sm70_tp4_hierarchical_reduce` and push kernel
`sm70_tp4_hierarchical_reduce_push`).

What this checks that tp4_hierarchical_correctness_test.py did not:
  * Every rank gets DIFFERENT input (seeded by rank). With identical inputs on
    all ranks, a kernel that mixed up ranks, cliques or the cross pair would
    still produce 4*x and pass.
  * Results are compared BIT-EXACTLY against the kernel's defined reduction
    tree, fp16((f32(r0) + f32(r1)) + (f32(r2) + f32(r3))), built from an
    NCCL all_gather of the raw inputs. NCCL's own sum is reported too, but
    only as a max-abs-diff (its order differs).
  * The push kernel only runs inside CUDA-graph capture, so correctness is
    exercised by replaying a captured graph that contains a MIXED sequence of
    message sizes. That stresses per-block epoch/slot reuse across size
    changes, which is where a push protocol would corrupt data or hang.
  * Timing is done inside CUDA graphs (K back-to-back all-reduces per replay)
    for custom AND NCCL, which is what decode actually pays. The older eager
    microbenchmark also timed a staging cudaMemcpy and host launch overhead.

Run (compare the two lines of output tables):
  VLLM_SM70_TP4_HIERARCHICAL_PUSH_AR=1 torchrun --nproc_per_node=4 \
      container_build/tp4_hierarchical_push_test.py
  VLLM_SM70_TP4_HIERARCHICAL_PUSH_AR=0 torchrun --nproc_per_node=4 \
      container_build/tp4_hierarchical_push_test.py

Optional:
  --sizes 5120,10240,40960     element counts to test (fp16)
  --replays 2000               stress replays of the mixed-size graph
  --k 64                       all-reduces per graph in the microbenchmark
  VLLM_SM70_TP4_HIERARCHICAL_PUSH_BLOCKS=N   sweep the push grid size
"""

import argparse
import os
import time

os.environ.setdefault("VLLM_SM70_TP4_HIERARCHICAL_CUSTOM_AR", "1")
os.environ.setdefault("VLLM_SM70_TP4_HIERARCHICAL_PUSH_AR", "1")
# Allow the whole kernel capacity during the test so every size is exercised.
os.environ.setdefault("VLLM_SM70_TP4_HIERARCHICAL_PUSH_MAX_BYTES", "163840")

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import (
    _SM70_TP4_HIERARCHICAL_ELEMENTS,
    CustomAllreduce,
)

# 1-token decode (5120), speculative verify rows (x2/x4/x5/x8/x16), and the
# two sizes the pull kernel already accepts.
DEFAULT_SIZES = [2560, 4096, 5120, 10240, 20480, 25600, 40960, 81920]


def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def tree_reference(gathered: list[torch.Tensor]) -> torch.Tensor:
    r0, r1, r2, r3 = (g.float() for g in gathered)
    return ((r0 + r1) + (r2 + r3)).half()


def fill_inputs(inputs, rank: int, seed: int) -> None:
    gen = torch.Generator(device="cuda")
    for i, x in enumerate(inputs):
        gen.manual_seed(seed * 100_003 + i * 1009 + rank)
        # Mix magnitudes so FP32 accumulation order actually matters.
        x.copy_(
            torch.randn(x.shape, generator=gen, device="cuda", dtype=torch.float32)
            * (10.0 ** (i % 3 - 1))
        )


def check(outputs, inputs, group) -> tuple[bool, float]:
    ok = True
    worst_nccl_diff = 0.0
    for out, inp in zip(outputs, inputs):
        gathered = [torch.empty_like(inp) for _ in range(4)]
        dist.all_gather(gathered, inp, group=group)
        expected = tree_reference(gathered)
        if not torch.equal(out, expected):
            ok = False
        nccl = inp.clone()
        dist.all_reduce(nccl, group=group)
        worst_nccl_diff = max(
            worst_nccl_diff, (out.float() - nccl.float()).abs().max().item()
        )
    return ok, worst_nccl_diff


def capture_graph(fn, k: int, ar: CustomAllreduce | None) -> torch.cuda.CUDAGraph:
    """Capture fn() k times. For custom AR, graph buffers are registered when
    ar.capture() exits, so the graph must not be replayed inside it."""
    g = torch.cuda.CUDAGraph()
    if ar is not None:
        with ar.capture(), torch.cuda.graph(g):
            for _ in range(k):
                fn()
    else:
        with torch.cuda.graph(g):
            for _ in range(k):
                fn()
    return g


def time_graph(g: torch.cuda.CUDAGraph, k: int, iters: int = 50) -> float:
    """Median per-call microseconds of a graph holding k collectives."""
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        dist.barrier()  # start all ranks together so skew is not timed
        start_evt.record()
        g.replay()
        end_evt.record()
        end_evt.synchronize()
        samples.append(start_evt.elapsed_time(end_evt) * 1000.0 / k)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default=None)
    parser.add_argument("--replays", type=int, default=2000)
    parser.add_argument("--k", type=int, default=64)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 4, f"TP4 test, got world_size={world_size}"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    nccl_group = dist.group.WORLD
    gloo_group = dist.new_group(backend="gloo")

    sizes = (
        [int(s) for s in args.sizes.split(",")] if args.sizes else DEFAULT_SIZES
    )

    ar = CustomAllreduce(group=gloo_group, device=local_rank)
    assert not ar.disabled, "CustomAllreduce disabled: check topology/env"
    assert ar.tp4_hierarchical, "2+2 hierarchical path did not activate"
    push_on = ar.sm70_tp4_hierarchical_push_buffer_ptrs is not None
    log(
        rank,
        f"push={'ON' if push_on else 'OFF'} "
        f"max_bytes={ar.tp4_hierarchical_push_max_bytes} "
        f"blocks={os.environ.get('VLLM_SM70_TP4_HIERARCHICAL_PUSH_BLOCKS', '40')} "
        f"sizes={sizes}",
    )
    # In-graph custom sizes: everything (push) or only the pull whitelist.
    graph_sizes = (
        sizes if push_on else [s for s in sizes if s in _SM70_TP4_HIERARCHICAL_ELEMENTS]
    )
    all_pass = True

    # --- 1. eager pull kernel, bit-exact (unchanged whitelist) -------------
    log(rank, "=== eager pull kernel (whitelist sizes), bit-exact ===")
    for numel in sorted(_SM70_TP4_HIERARCHICAL_ELEMENTS):
        inp = torch.empty(numel, dtype=torch.float16, device="cuda")
        bad = 0
        for seed in range(50):
            fill_inputs([inp], rank, seed)
            out = ar.custom_all_reduce(inp)
            assert out is not None, f"eager path rejected numel={numel}"
            ok, _ = check([out], [inp], nccl_group)
            bad += 0 if ok else 1
        all_pass &= bad == 0
        log(rank, f"  numel={numel:6d}: {50 - bad}/50 bit-exact")

    # --- 2. mixed-size graph: correctness + stress --------------------------
    if graph_sizes:
        log(rank, "=== mixed-size CUDA graph, bit-exact vs reduction tree ===")
        # Interleave sizes (big, small, big, ...) so consecutive calls change
        # size, then repeat so each size is hit several times per replay.
        order = []
        for _ in range(3):
            order += sorted(graph_sizes, reverse=True)
            order += sorted(graph_sizes)
        inputs = [torch.empty(n, dtype=torch.float16, device="cuda") for n in order]
        # Capture directly: non-whitelist sizes route to NCCL when eager, so
        # there is no meaningful eager warm-up for them.
        outputs: list[torch.Tensor | None] = []
        g = torch.cuda.CUDAGraph()
        with ar.capture(), torch.cuda.graph(g):
            outputs.extend(ar.custom_all_reduce(x) for x in inputs)
        assert all(o is not None for o in outputs), "capture rejected a size"

        failures = 0
        worst = 0.0
        checked = 0
        t0 = time.time()
        for i in range(args.replays):
            fill_inputs(inputs, rank, seed=i)
            g.replay()
            if i % 25 == 0 or i == args.replays - 1:
                torch.cuda.synchronize()
                ok, nccl_diff = check(outputs, inputs, nccl_group)
                checked += 1
                worst = max(worst, nccl_diff)
                if not ok:
                    failures += 1
                    print(f"[rank {rank}] MISMATCH at replay {i}", flush=True)
        torch.cuda.synchronize()
        all_pass &= failures == 0
        log(
            rank,
            f"  {args.replays} replays x {len(order)} all-reduces in "
            f"{time.time() - t0:.1f}s, no hang; {checked - failures}/{checked} "
            f"checked replays bit-exact; max |custom - NCCL| = {worst:.4g}",
        )

    # --- 3. bf16 bypass -------------------------------------------------------
    dist.barrier()
    bf16_out = ar.custom_all_reduce(
        torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    )
    all_pass &= bf16_out is None
    log(rank, f"=== bf16 bypass: {'ok' if bf16_out is None else 'FAILED'} ===")

    # --- 4. in-graph microbenchmark -----------------------------------------
    dist.barrier()
    log(rank, f"=== in-graph microbenchmark (median us/call, {args.k} per graph) ===")
    log(rank, f"  {'numel':>7} {'bytes':>7} {'custom':>9} {'nccl':>9} {'speedup':>8}")
    for numel in sizes:
        inp = torch.randn(numel, dtype=torch.float16, device="cuda")
        custom_us = float("nan")
        if numel in graph_sizes:
            g = capture_graph(lambda: ar.custom_all_reduce(inp), args.k, ar)
            custom_us = time_graph(g, args.k)
        nccl_buf = inp.clone()
        try:
            g = capture_graph(
                lambda: dist.all_reduce(nccl_buf, group=nccl_group), args.k, None
            )
            nccl_us = time_graph(g, args.k)
        except Exception as exc:  # NCCL capture unsupported in this build
            nccl_us = float("nan")
            log(rank, f"  (NCCL graph capture failed: {exc})")
        log(
            rank,
            f"  {numel:7d} {numel * 2:7d} {custom_us:9.2f} {nccl_us:9.2f} "
            f"{nccl_us / custom_us:7.2f}x",
        )

    dist.barrier()
    log(rank, "\n=== RESULT ===")
    log(rank, "ALL CHECKS PASSED" if all_pass else "*** FAILURE DETECTED ABOVE ***")
    ar.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
