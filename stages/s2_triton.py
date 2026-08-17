"""Stage 2: does Triton work here, and what does it cost.

Every case runs in its own subprocess. That is not defensive padding: a
non-NVIDIA Triton backend tested earlier did not raise on a perfectly ordinary
`range()` loop carrying a tile through a reduction, it killed the process with
SIGSEGV, and a driver that does that would otherwise take the whole survey down
with it. A crash here is recorded as a finding and the next case still runs.

The cases are ordered so the cheap yes/no questions come before the
measurements, because if `add` cannot compile then nothing after it means
anything.

    python3 stages/s2_triton.py                 # all cases
    python3 stages/s2_triton.py --case dot      # one case, in this process
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ks_device  # noqa: E402

CASES = (
    ("add", "compile and run the simplest possible kernel"),
    ("launch", "per-launch cost: Triton vs torch vs allocation"),
    ("fastlaunch", "can CompiledKernel.run be called directly"),
    ("bandwidth", "peak copy bandwidth from a Triton kernel"),
    ("reduce2d", "reductions over a 2D tile, both axes"),
    ("loop_reduce", "reduction inside range() -- a known backend crash pattern"),
    ("static8", "tl.static_range unroll of 8 rounds, and its compile time"),
    ("static16", "tl.static_range unroll of 16 rounds, and its compile time"),
    ("tile3d", "3D tile carrying a reduction"),
    ("dot", "tl.dot against the vendor BLAS on the SPLADE shape"),
    ("dtypes", "elementwise throughput per dtype"),
    ("grid_scaling", "how many programs run concurrently -- the AI core count"),
    ("tile_limit", "largest tile a single program can hold"),
    ("num_warps", "whether num_warps means anything on this backend"),
)


def _kernels():
    """Built lazily so a case can report an import failure instead of crashing.

    The `global` matters. Triton does not compile the Python function object it
    was handed; it re-parses the source and resolves every name -- including the
    `tl` in a `BLOCK: tl.constexpr` annotation -- against the defining module's
    globals. A plain `import triton.language as tl` here would bind `tl` as a
    local of this function, leaving it absent from those globals, and every
    kernel below would fail to compile with NameError('tl is not defined').
    """
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def k_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)

    @triton.jit
    def k_copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m), mask=m)

    @triton.jit
    def k_reduce2d(x_ptr, r_ptr, c_ptr, R: tl.constexpr, C: tl.constexpr):
        r = tl.arange(0, R)
        c = tl.arange(0, C)
        v = tl.load(x_ptr + r[:, None] * C + c[None, :])
        tl.store(r_ptr + r, tl.sum(v, axis=1))
        tl.store(c_ptr + c, tl.sum(v, axis=0))

    @triton.jit
    def k_loop_reduce(x_ptr, out_ptr, ITERS: tl.constexpr, HC: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, HC)
        off = c[:, None] * HC + c[None, :]
        v = tl.load(x_ptr + row * HC * HC + off)
        for _ in range(ITERS):
            v = v / (tl.sum(v, axis=1)[:, None] + 1e-6)
            v = v / (tl.sum(v, axis=0)[None, :] + 1e-6)
        tl.store(out_ptr + row * HC * HC + off, v)

    @triton.jit
    def k_static(x_ptr, out_ptr, ITERS: tl.constexpr, HC: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, HC)
        off = c[:, None] * HC + c[None, :]
        v = tl.load(x_ptr + row * HC * HC + off)
        for _ in tl.static_range(ITERS):
            v = v / (tl.sum(v, axis=1)[:, None] + 1e-6)
            v = v / (tl.sum(v, axis=0)[None, :] + 1e-6)
        tl.store(out_ptr + row * HC * HC + off, v)

    @triton.jit
    def k_tile3d(x_ptr, out_ptr, A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
        a = tl.arange(0, A)
        b = tl.arange(0, B)
        c = tl.arange(0, C)
        off = a[:, None, None] * (B * C) + b[None, :, None] * C + c[None, None, :]
        v = tl.load(x_ptr + off)
        tl.store(out_ptr + a[:, None] * C + c[None, :], tl.sum(v, axis=1))

    @triton.jit
    def k_dot(a_ptr, b_ptr, out_ptr, M, N, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(a_ptr + rm[:, None] * K + rk[None, :],
                        mask=(rm[:, None] < M), other=0.0)
            b = tl.load(b_ptr + rk[:, None] * N + rn[None, :],
                        mask=(rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    return dict(add=k_add, copy=k_copy, reduce2d=k_reduce2d,
                loop_reduce=k_loop_reduce, static=k_static,
                tile3d=k_tile3d, dot=k_dot)


def case_add(K):
    import torch
    dev = ks_device.device_str()
    n = 4096
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)
    K["add"][(1,)](x, y, out, n, BLOCK=4096)
    ks_device.synchronize()
    err = (out - (x + y)).abs().max().item()
    print("  triton compiles and runs: yes, max err {:.2e}".format(err))


def case_launch(K):
    import torch
    dev = ks_device.device_str()
    n = 1024
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)
    K["add"][(1,)](x, y, out, n, BLOCK=1024)
    ks_device.synchronize()

    tri = ks_device.bench(lambda: K["add"][(1,)](x, y, out, n, BLOCK=1024))
    tor = ks_device.bench(lambda: torch.mul(x, 2.0, out=out))
    alloc = ks_device.bench(lambda: torch.empty(n, device=dev))
    empty = ks_device.bench(lambda: None)
    print("  triton kernel[grid](...)     {:8.2f} us".format(tri * 1e3))
    print("  torch elementwise            {:8.2f} us".format(tor * 1e3))
    print("  torch.empty(1024)            {:8.2f} us".format(alloc * 1e3))
    print("  synchronize floor            {:8.2f} us".format(empty * 1e3))


def case_fastlaunch(K):
    import torch
    dev = ks_device.device_str()
    n = 1024
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)

    compiled = K["add"][(1,)](x, y, out, n, BLOCK=1024)
    ks_device.synchronize()
    if compiled is None:
        print("  kernel[grid](...) returned None; cannot reach the compiled object")
        return
    print("  compiled object: {}".format(type(compiled).__name__))
    for attr in ("run", "function", "packed_metadata", "metadata", "module"):
        print("    has {:<18} {}".format(attr, hasattr(compiled, attr)))

    import inspect
    print("  run signature: {}".format(inspect.signature(compiled.run)))

    stream = _raw_stream()
    print("  raw stream handle: {}".format(
        "obtained" if stream is not None else "NOT available"))
    if stream is None:
        print("  without a stream handle the _FastLaunch port cannot be tried")
        return

    # The 9-element prefix used by CUDA-style backends. If this one orders
    # CompiledKernel.run's arguments differently it fails here, which is exactly
    # the thing worth knowing.
    try:
        compiled.run(1, 1, 1, stream, compiled.function, compiled.packed_metadata,
                     None, None, None, x, y, out, n)
        ks_device.synchronize()
        err = (out - (x + y)).abs().max().item()
        print("  direct run, CUDA 9-arg prefix: works, max err {:.2e}".format(err))
    except Exception as exc:
        print("  direct run, CUDA 9-arg prefix: {}: {}".format(
            type(exc).__name__, str(exc)[:200]))


def _raw_stream():
    """The backend's current stream as whatever integer handle it exposes."""
    mod = ks_device.accel_module()
    if mod is None:
        return None
    try:
        stream = mod.current_stream()
    except Exception:
        return None
    for attr in ("cuda_stream", "npu_stream", "_as_parameter_", "id"):
        handle = getattr(stream, attr, None)
        if isinstance(handle, int):
            return handle
    return None


def case_bandwidth(K):
    import torch
    dev = ks_device.device_str()
    for dtype, label in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        try:
            n = 32 * 1024 * 1024
            x = torch.randn(n, device=dev, dtype=dtype)
            out = torch.empty_like(x)
            block = 4096
            grid = ((n + block - 1) // block,)
            K["copy"][grid](x, out, n, BLOCK=block)
            ks_device.synchronize()
            ms = ks_device.bench(lambda: K["copy"][grid](x, out, n, BLOCK=block),
                                 warmup=5, repeat=25)
            gb = 2 * x.numel() * x.element_size() / 1e9
            print("  copy {:<9} {:8.3f} ms  {:7.0f} GB/s".format(label, ms, gb / (ms * 1e-3)))
            del x, out
        except Exception as exc:
            print("  copy {:<9} {}: {}".format(label, type(exc).__name__, str(exc)[:120]))


def case_reduce2d(K):
    import torch
    dev = ks_device.device_str()
    R = C = 64
    x = torch.randn(R * C, device=dev)
    r = torch.empty(R, device=dev)
    c = torch.empty(C, device=dev)
    K["reduce2d"][(1,)](x, r, c, R=R, C=C)
    ks_device.synchronize()
    xm = x.view(R, C)
    er = (r - xm.sum(1)).abs().max().item()
    ec = (c - xm.sum(0)).abs().max().item()
    print("  row-sum err {:.2e}, col-sum err {:.2e}".format(er, ec))


def case_loop_reduce(K):
    import torch
    dev = ks_device.device_str()
    hc, rows = 4, 16
    x = torch.rand(rows * hc * hc, device=dev) + 0.5
    out = torch.empty_like(x)
    K["loop_reduce"][(rows,)](x, out, ITERS=19, HC=hc)
    ks_device.synchronize()
    print("  19 rounds inside range(): survived, finite={}".format(
        bool(out.isfinite().all().item())))


def _case_static(K, iters):
    import torch
    dev = ks_device.device_str()
    hc, rows = 4, 16
    x = torch.rand(rows * hc * hc, device=dev) + 0.5
    out = torch.empty_like(x)
    t0 = time.perf_counter()
    K["static"][(rows,)](x, out, ITERS=iters, HC=hc)
    ks_device.synchronize()
    print("  {} rounds unrolled: compiled and ran in {:.1f} s".format(
        iters, time.perf_counter() - t0))


def case_static8(K):
    _case_static(K, 8)


def case_static16(K):
    _case_static(K, 16)


def case_tile3d(K):
    import torch
    dev = ks_device.device_str()
    A, B, C = 4, 8, 16
    x = torch.randn(A * B * C, device=dev)
    out = torch.empty(A * C, device=dev)
    K["tile3d"][(1,)](x, out, A=A, B=B, C=C)
    ks_device.synchronize()
    err = (out.view(A, C) - x.view(A, B, C).sum(1)).abs().max().item()
    print("  3D tile with reduction: works, max err {:.2e}".format(err))


def case_dot(K):
    import torch
    import torch.nn.functional as F
    dev = ks_device.device_str()
    # The shape SPLADESparsePooler actually needs.
    M, Kd, N = 83, 768, 30522
    for dtype, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        try:
            a = torch.randn(M, Kd, device=dev, dtype=dtype)
            b = torch.randn(Kd, N, device=dev, dtype=dtype)
            w = b.t().contiguous()
            ms = ks_device.bench(lambda: a @ b, warmup=10, repeat=30)
            ms_lin = ks_device.bench(lambda: F.linear(a, w), warmup=10, repeat=30)
            mac = M * Kd * N
            print("  torch a@b     {:<5} {:8.1f} us  {:6.2f} TMAC/s".format(
                label, ms * 1e3, mac / (ms * 1e-3) / 1e12))
            print("  torch F.linear{:<5} {:8.1f} us  {:6.2f} TMAC/s".format(
                label, ms_lin * 1e3, mac / (ms_lin * 1e-3) / 1e12))
            del a, b, w
        except Exception as exc:
            print("  torch matmul  {:<5} {}: {}".format(label, type(exc).__name__, str(exc)[:110]))

    for (bm, bn, bk) in ((64, 64, 32), (32, 128, 32), (16, 64, 32)):
        try:
            a = torch.randn(M, Kd, device=dev, dtype=torch.float16)
            b = torch.randn(Kd, N, device=dev, dtype=torch.float16)
            out = torch.empty(M, N, device=dev, dtype=torch.float32)
            grid = ((M + bm - 1) // bm, (N + bn - 1) // bn)
            K["dot"][grid](a, b, out, M, N, K=Kd, BM=bm, BN=bn, BK=bk)
            ks_device.synchronize()
            err = (out - (a.float() @ b.float())).abs().max().item()
            ms = ks_device.bench(
                lambda: K["dot"][grid](a, b, out, M, N, K=Kd, BM=bm, BN=bn, BK=bk),
                warmup=5, repeat=20)
            print("  tl.dot {}x{}x{}  {:8.1f} us  {:6.2f} TMAC/s  err={:.1e}".format(
                bm, bn, bk, ms * 1e3, M * Kd * N / (ms * 1e-3) / 1e12, err))
            del a, b, out
        except Exception as exc:
            print("  tl.dot {}x{}x{}  {}: {}".format(
                bm, bn, bk, type(exc).__name__, str(exc).splitlines()[0][:110]))


def case_dtypes(K):
    import torch
    dev = ks_device.device_str()
    n = 4 * 1024 * 1024
    for dtype, label in ((torch.float32, "float32"), (torch.float16, "float16"),
                         (torch.bfloat16, "bfloat16")):
        try:
            x = torch.randn(n, device=dev, dtype=dtype)
            y = torch.randn(n, device=dev, dtype=dtype)
            out = torch.empty_like(x)
            block = 4096
            grid = ((n + block - 1) // block,)
            K["add"][grid](x, y, out, n, BLOCK=block)
            ks_device.synchronize()
            ms = ks_device.bench(lambda: K["add"][grid](x, y, out, n, BLOCK=block),
                                 warmup=10, repeat=30)
            print("  elementwise {:<9} {:8.1f} us".format(label, ms * 1e3))
            del x, y, out
        except Exception as exc:
            print("  elementwise {:<9} {}: {}".format(label, type(exc).__name__, str(exc)[:110]))


def case_grid_scaling(K):
    """Find how many programs actually run at once.

    Each program does the same fixed amount of work, so while the grid fits in
    hardware the wall time barely moves; once it does not, time grows in steps.
    The knee is the number of concurrently executing programs.

    This is the single most important number for porting to this architecture.
    A DaVinci part has tens of AI cores, not thousands of CUDA-style SMs, so
    the usual "one program per small tile, let the scheduler sort it out"
    decomposition behaves completely differently here, and every block-size
    choice in the ten operators depends on knowing where the knee is.
    """
    import torch
    dev = ks_device.device_str()
    block = 1024
    print("  {:>7}  {:>10}  {:>12}".format("grid", "ms", "GB/s"))
    for grid in (1, 2, 4, 8, 16, 20, 24, 32, 40, 48, 64, 96, 128, 256, 1024, 4096):
        try:
            n = grid * block
            x = torch.randn(n, device=dev)
            out = torch.empty_like(x)
            K["copy"][(grid,)](x, out, n, BLOCK=block)
            ks_device.synchronize()
            ms = ks_device.bench(lambda: K["copy"][(grid,)](x, out, n, BLOCK=block),
                                 warmup=10, repeat=40)
            gb = 2 * x.numel() * x.element_size() / 1e9
            print("  {:>7}  {:>10.4f}  {:>12.1f}".format(grid, ms, gb / (ms * 1e-3)))
            del x, out
        except Exception as exc:
            print("  {:>7}  {}: {}".format(grid, type(exc).__name__, str(exc)[:90]))


def case_tile_limit(K):
    """Largest tile one program can hold before the compiler or runtime refuses.

    On this architecture that ceiling is the on-core buffer, and it decides how
    much of an operator can stay resident instead of being split across extra
    launches. Hitting such a limit usually means restructuring an operator into
    several launches, so it is worth knowing before writing anything.
    """
    import torch
    dev = ks_device.device_str()
    largest = None
    for block in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
        try:
            x = torch.randn(block, device=dev)
            out = torch.empty(1, device=dev)
            K["reduce2d"][(1,)](x, out, out, R=1, C=block)
            ks_device.synchronize()
            largest = block
            print("  fp32 tile of {:>7} elements ({:>5} KB): ok".format(
                block, block * 4 // 1024))
            del x, out
        except Exception as exc:
            print("  fp32 tile of {:>7} elements ({:>5} KB): {}: {}".format(
                block, block * 4 // 1024, type(exc).__name__,
                str(exc).splitlines()[0][:90]))
            break
    print("  largest working tile: {}".format(largest))


def case_num_warps(K):
    """Does num_warps do anything here.

    Kernels carried over from a CUDA-style backend usually pass an explicit
    num_warps tuned for that hardware. If this backend ignores the argument
    those numbers are dead weight; if it rejects it, they are a compile error in
    every operator that sets one.
    """
    import torch
    dev = ks_device.device_str()
    n = 1024 * 1024
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    out = torch.empty_like(x)
    block = 1024
    grid = (n // block,)
    for nw in (1, 2, 4, 8, 16):
        try:
            K["add"][grid](x, y, out, n, BLOCK=block, num_warps=nw)
            ks_device.synchronize()
            ms = ks_device.bench(
                lambda w=nw: K["add"][grid](x, y, out, n, BLOCK=block, num_warps=w),
                warmup=10, repeat=30)
            print("  num_warps={:<3} {:8.1f} us".format(nw, ms * 1e3))
        except Exception as exc:
            print("  num_warps={:<3} {}: {}".format(
                nw, type(exc).__name__, str(exc).splitlines()[0][:100]))


def run_one(name):
    K = _kernels()
    globals()["case_" + name](K)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    ks_device.require_accel()

    if args.case:
        run_one(args.case)
        return 0

    print("device: {} ({} visible)".format(ks_device.device_name(), ks_device.device_count()))
    for name, blurb in CASES:
        print("\n--- {} : {} ---".format(name, blurb))
        sys.stdout.flush()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--case", name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT: still running after {} s".format(args.timeout))
            continue
        text = proc.stdout.decode("utf-8", "replace").rstrip()
        if text:
            print(text)
        if proc.returncode != 0:
            if proc.returncode == -11:
                print("  *** SIGSEGV: this construct crashes the backend ***")
            else:
                print("  *** exited with {} ***".format(proc.returncode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
