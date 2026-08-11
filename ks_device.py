"""Accelerator abstraction shared by the survey stages.

Everything else in this bundle avoids naming a backend directly. On Ascend the
device is 'npu' and only exists as torch.npu after torch_npu is imported, which
sitecustomize.py handles; on a CUDA box the same code runs unchanged, which
matters because it lets the bundle be smoke-tested somewhere else before it is
sent to a machine nobody can log into.

Timing is deliberately synchronize + perf_counter rather than device events:
that is what the official auto_bench.py does, so measurements here are
comparable with the scores, and it does not depend on any backend having an
event API.
"""

import statistics
import time

import torch

# Order matters only in that the first available one wins; a box has one.
_CANDIDATES = ("npu", "cuda", "mlu", "gcu")


def accel_name():
    """Name of the first available accelerator, or None if this is CPU only."""
    for name in _CANDIDATES:
        mod = getattr(torch, name, None)
        if mod is None:
            continue
        try:
            if mod.is_available():
                return name
        except Exception:
            continue
    return None


def accel_module(name=None):
    name = name or accel_name()
    return getattr(torch, name) if name else None


def device_str(index=0):
    name = accel_name()
    if name is None:
        raise RuntimeError("no accelerator available; this bundle needs one")
    return "{}:{}".format(name, index)


def synchronize():
    mod = accel_module()
    if mod is not None:
        try:
            mod.synchronize()
        except Exception:
            pass


def device_name(index=0):
    mod = accel_module()
    if mod is None:
        return "cpu"
    for attr in ("get_device_name", "get_device_properties"):
        fn = getattr(mod, attr, None)
        if fn is None:
            continue
        try:
            got = fn(index)
        except Exception:
            continue
        return got if isinstance(got, str) else getattr(got, "name", str(got))
    return accel_name()


def device_count():
    mod = accel_module()
    if mod is None:
        return 0
    try:
        return mod.device_count()
    except Exception:
        return 0


def require_accel():
    """Accelerator name, or exit with one clear line rather than a traceback.

    Used at the top of each stage so that running one standalone on a machine
    whose driver is unhappy says so once, instead of raising the same error out
    of every case.
    """
    import sys
    name = accel_name()
    if name is None:
        sys.stderr.write(
            "no accelerator is usable from Python (torch sees no npu/cuda/mlu/gcu).\n"
            "On Ascend, source the CANN environment first:\n"
            "    source /usr/local/Ascend/ascend-toolkit/set_env.sh\n")
        raise SystemExit(2)
    return name


def bench(fn, warmup=25, repeat=100):
    """Median wall-clock milliseconds per call, synchronising around each one.

    Median rather than mean because a stray scheduling hiccup on a shared
    machine otherwise dominates, and the scorer takes a median too.
    """
    for _ in range(warmup):
        fn()
    synchronize()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)
