"""Stage 5: what a launch actually costs, and whether launches overlap.

Stage 2 established that one `kernel[grid](...)` costs about 139 us end to end
while the synchronise floor is 17 us, which makes dispatch the dominant cost for
most of these operators. It left two questions open, and between them they
decide how half the entries have to be written:

  1. Does calling CompiledKernel.run directly cost less than going through the
     JIT dispatcher? Stage 2 proved the call works and produces correct results
     but never timed it.

  2. Do consecutive launches overlap? If issuing eight kernels and
     synchronising once costs about as much as issuing one, dispatch is
     asynchronous and a multi-kernel design is fine. If it costs eight times as
     much, dispatch is serial host work and every operator has to collapse into
     a single kernel no matter how awkward that is.

The kernel used here is deliberately trivial and operates on 1024 elements, so
device time is negligible and what is left is overhead.

    python3 stages/s5_launch.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ks_device  # noqa: E402

N = 1024


def _kernels():
    # Same `global` requirement as stage 2: Triton re-parses the source and
    # resolves `tl` against module globals, not this function's locals.
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def k_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)

    return k_add


def _raw_stream():
    mod = ks_device.accel_module()
    try:
        stream = mod.current_stream()
    except Exception:
        return None
    for attr in ("npu_stream", "cuda_stream", "_as_parameter_", "id"):
        handle = getattr(stream, attr, None)
        if isinstance(handle, int):
            return handle
    return None


def timed(fn, warmup=20, repeat=60):
    for _ in range(warmup):
        fn()
    ks_device.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        ks_device.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples) * 1000.0  # microseconds


def main():
    import torch

    ks_device.require_accel()
    dev = ks_device.device_str()
    k_add = _kernels()

    x = torch.randn(N, device=dev)
    y = torch.randn(N, device=dev)
    out = torch.empty_like(x)

    compiled = k_add[(1,)](x, y, out, N, BLOCK=N)
    ks_device.synchronize()
    stream = _raw_stream()

    print("device: {}".format(ks_device.device_name()))
    print("\n--- floor ---")
    print("  synchronize only            {:8.1f} us".format(timed(lambda: None)))

    print("\n--- one launch, one sync ---")
    jit_us = timed(lambda: k_add[(1,)](x, y, out, N, BLOCK=N))
    print("  kernel[grid](...)           {:8.1f} us".format(jit_us))

    fast_us = None
    if stream is None:
        print("  CompiledKernel.run          no stream handle; cannot test")
    else:
        def fast():
            compiled.run(1, 1, 1, stream, compiled.function,
                         compiled.packed_metadata, None, None, None, x, y, out, N)
        try:
            fast()
            ks_device.synchronize()
            fast_us = timed(fast)
            print("  CompiledKernel.run direct   {:8.1f} us   ({:.0f}% of the dispatcher)".format(
                fast_us, 100.0 * fast_us / jit_us))
        except Exception as exc:
            print("  CompiledKernel.run          {}: {}".format(
                type(exc).__name__, str(exc)[:120]))

    print("\n--- n launches, one sync: does dispatch overlap ---")
    print("  {:>3}  {:>12}  {:>12}  {:>10}".format("n", "total us", "per launch", "verdict"))
    for n in (1, 2, 4, 8, 16):
        def many(count=n):
            for _ in range(count):
                k_add[(1,)](x, y, out, N, BLOCK=N)
        total = timed(many, warmup=5, repeat=20)
        print("  {:>3}  {:>12.1f}  {:>12.1f}".format(n, total, total / n))

    if fast_us is not None:
        print("\n  same, through the direct path:")
        for n in (1, 2, 4, 8, 16):
            def many_fast(count=n):
                for _ in range(count):
                    compiled.run(1, 1, 1, stream, compiled.function,
                                 compiled.packed_metadata, None, None, None,
                                 x, y, out, N)
            total = timed(many_fast, warmup=5, repeat=20)
            print("  {:>3}  {:>12.1f}  {:>12.1f}".format(n, total, total / n))

    print("\n--- n torch ops, one sync: the reference implementations' cost model ---")
    print("  {:>3}  {:>12}  {:>12}".format("n", "total us", "per op"))
    for n in (1, 2, 4, 8, 16):
        def many_torch(count=n):
            for _ in range(count):
                torch.mul(x, 2.0, out=out)
        total = timed(many_torch, warmup=5, repeat=20)
        print("  {:>3}  {:>12.1f}  {:>12.1f}".format(n, total, total / n))

    print("\n--- cost of a device-to-host sync, as the references pay it ---")
    flags = torch.ones(8, dtype=torch.bool, device=dev)
    print("  tensor.any() -> python bool {:8.1f} us".format(
        timed(lambda: bool(flags.any()))))
    print("  tensor[mask] boolean index  {:8.1f} us".format(
        timed(lambda: x[:8][flags])))
    print("  .item() on a scalar         {:8.1f} us".format(
        timed(lambda: x[0].item())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
