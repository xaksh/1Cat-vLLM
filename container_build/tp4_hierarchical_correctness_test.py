"""
Standalone correctness + microbenchmark harness for the new TP4 hierarchical
custom allreduce (sm70_tp4_hierarchical_reduce), run outside the full vLLM
engine.

For each of the 5 accepted element sizes, across many random seeds:
  1. Build identical random fp16 input on all 4 ranks.
  2. Run CustomAllreduce with VLLM_SM70_TP4_HIERARCHICAL_CUSTOM_AR=1 (routes
     into our new kernel since world_size=4 and this box is not fully
     NVLink-connected).
  3. Run plain NCCL dist.all_reduce on an independent copy of the same input.
  4. Compare with torch.allclose (fp16 tolerance -- exact bitwise equality is
     NOT expected since our clique-then-cross-pair summation order differs
     from NCCL's ring/tree order).
  5. Also confirms bf16 input bypasses the new path (should fall through to
     NCCL, not crash / not silently misroute).

Also times both paths (median over many iterations) as a lightweight
microbenchmark.

Run with:
  torchrun --nproc_per_node=4 tp4_hierarchical_correctness_test.py
"""

import os

os.environ.setdefault("VLLM_SM70_TP4_HIERARCHICAL_CUSTOM_AR", "1")

import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    _SM70_TP4_HIERARCHICAL_ELEMENTS,
)

SEEDS_PER_SIZE = 100
STRESS_ITERS = 2000
WARMUP_ITERS = 10
TIMED_ITERS = 200


def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 4, f"This test is specifically for TP4, got world_size={world_size}"

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    # CustomAllreduce asserts it is attached to a non-NCCL group (it does its
    # own IPC/CUDA-graph coordination); use a separate gloo group for it while
    # keeping the default NCCL group for the plain dist.all_reduce baseline.
    gloo_group = dist.new_group(backend="gloo")

    ar = CustomAllreduce(group=gloo_group, device=local_rank)
    log(rank, f"CustomAllreduce constructed: disabled={ar.disabled} tp4_hierarchical={getattr(ar, 'tp4_hierarchical', None)} fully_connected={getattr(ar, 'fully_connected', None)}")

    assert not ar.disabled, "CustomAllreduce is disabled -- check topology/env var/device capability"
    assert ar.tp4_hierarchical, (
        "tp4_hierarchical is False -- the new path did not activate. "
        "Check VLLM_SM70_TP4_HIERARCHICAL_CUSTOM_AR, world_size, and is_fully_connected()."
    )

    sizes = sorted(_SM70_TP4_HIERARCHICAL_ELEMENTS)
    log(rank, f"Testing element sizes: {sizes}")

    all_pass = True
    for numel in sizes:
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        for seed in range(SEEDS_PER_SIZE):
            torch.manual_seed(seed * 1000 + numel)
            local_input = torch.randn(numel, dtype=torch.float16, device="cuda")
            # Independent copies for the two paths (all_reduce mutates via
            # out-of-place semantics here, but keep them clearly separate).
            custom_input = local_input.clone()
            nccl_input = local_input.clone()

            custom_out = ar.custom_all_reduce(custom_input)
            assert custom_out is not None, (
                f"custom_all_reduce returned None for numel={numel} (should_custom_ar "
                "rejected a size we expected to be accepted -- check the element/byte gate)"
            )

            dist.all_reduce(nccl_input, group=group)

            diff = (custom_out.float() - nccl_input.float()).abs()
            abs_diff = diff.max().item()
            denom = nccl_input.float().abs().clamp_min(1e-3)
            rel_diff = (diff / denom).max().item()
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)

            ok = torch.allclose(custom_out.float(), nccl_input.float(), rtol=1e-2, atol=1e-2)
            if not ok:
                all_pass = False
                print(
                    f"[rank {rank}] MISMATCH numel={numel} seed={seed} "
                    f"abs_diff={abs_diff:.6f} rel_diff={rel_diff:.6f}",
                    flush=True,
                )

        log(rank, f"numel={numel:6d}: max_abs_diff={max_abs_diff:.6f} max_rel_diff={max_rel_diff:.6f} over {SEEDS_PER_SIZE} seeds")

    dist.barrier()
    log(rank, "=== bf16 bypass check ===")
    bf16_input = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    bf16_out = ar.custom_all_reduce(bf16_input)
    assert bf16_out is None, "bf16 input should NOT be routed into the fp16-only hierarchical kernel"
    log(rank, "bf16 correctly bypassed the custom path (returned None, falls through to NCCL)")

    dist.barrier()
    log(rank, "=== stress test (checking for hangs / late corruption) ===")
    stress_numel = 4096
    stress_start = time.time()
    for i in range(STRESS_ITERS):
        torch.manual_seed(i)
        inp = torch.randn(stress_numel, dtype=torch.float16, device="cuda")
        ref = inp.clone()
        out = ar.custom_all_reduce(inp)
        dist.all_reduce(ref, group=group)
        if not torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2):
            all_pass = False
            print(f"[rank {rank}] STRESS MISMATCH at iter {i}", flush=True)
            break
    else:
        log(rank, f"stress test: {STRESS_ITERS} iterations completed, {time.time() - stress_start:.1f}s, no hang, no mismatch")

    dist.barrier()
    log(rank, "=== microbenchmark ===")
    for numel in sizes:
        inp = torch.randn(numel, dtype=torch.float16, device="cuda")
        for _ in range(WARMUP_ITERS):
            ar.custom_all_reduce(inp)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            ar.custom_all_reduce(inp)
        torch.cuda.synchronize()
        custom_us = (time.perf_counter() - start) / TIMED_ITERS * 1e6

        for _ in range(WARMUP_ITERS):
            dist.all_reduce(inp, group=group)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            dist.all_reduce(inp, group=group)
        torch.cuda.synchronize()
        nccl_us = (time.perf_counter() - start) / TIMED_ITERS * 1e6

        log(rank, f"numel={numel:6d}  custom={custom_us:8.2f}us  nccl={nccl_us:8.2f}us  speedup={nccl_us / custom_us:.2f}x")

    dist.barrier()
    if rank == 0:
        print("\n=== RESULT ===", flush=True)
        print("ALL CHECKS PASSED" if all_pass else "*** CORRECTNESS FAILURE DETECTED ABOVE ***", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
