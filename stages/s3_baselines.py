"""Stage 3: reference latency and input layout for each of the ten operators.

Two things come out of this. The obvious one is v0_ms per operator, which is the
denominator of every score and says where the absolute headroom is. The less
obvious one is the exact shape, dtype, stride and contiguity of every input as
the operator actually receives it on this device.

That second part earns its keep. On a backend tested earlier, moving a 4-D
tensor to the device silently changed its memory format; a kernel's guard then
rejected the unexpected strides and the operator quietly scored 1.00x by
falling back to the reference. Nothing about that failure was visible in a
latency table. Reporting strides here catches the whole class of surprise before
any kernel is written for this machine.

Each operator runs in its own subprocess, for the same reason as stage 2.

    python3 stages/s3_baselines.py
    python3 stages/s3_baselines.py --task mhc_post
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ks_device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "tasks")
BASELINES = os.path.join(ROOT, "baselines")

# Ordered by the absolute time the reference spends, largest first, as measured
# on other hardware. The biggest baselines are where a given
# percentage of improvement is worth the most, so if the run has to be cut short
# the useful numbers are already in.
ORDER = (
    "mhc_post",
    "FusedMoE",
    "hc_split_sinkhorn",
    "CentreRandomAugmentation",
    "SPLADESparsePooler",
    "GroupedTopk",
    "MusicFlamingoRotaryEmbedding",
    "head_compute_mix_bwd",
    "FlexAttention",
    "MMEncoderAttention",
)


def prepare():
    """Turn tasks/*.txt into runnable baselines/*.py for this accelerator.

    The competition sources hardcode device="cuda" in three of the ten
    get_inputs(). auto_bench.py rewrites device literals for other backends but
    returns early on Ascend, so on an NPU those three would fail outright. The
    literal is swapped here instead, for the reference and our own entries
    alike, so the comparison stays apples to apples.
    """
    accel = ks_device.require_accel()
    os.makedirs(BASELINES, exist_ok=True)
    made = []
    for name in sorted(os.listdir(TASKS)):
        if not name.endswith(".txt"):
            continue
        src = open(os.path.join(TASKS, name), encoding="utf-8").read()
        if accel != "cuda":
            src = re.sub(r'(["\'])cuda\1', r'\g<1>{}\g<1>'.format(accel), src)
            src = src.replace(".cuda()", '.to("{}")'.format(accel))
        out = os.path.join(BASELINES, name[:-4] + ".py")
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(src)
        made.append(name[:-4])
    return made


def describe(value, prefix=""):
    import torch
    lines = []
    if isinstance(value, torch.Tensor):
        fmts = []
        if value.is_contiguous():
            fmts.append("contiguous")
        for fname, fmt in (("channels_last", getattr(torch, "channels_last", None)),
                           ("channels_last_3d", getattr(torch, "channels_last_3d", None))):
            if fmt is None:
                continue
            try:
                if value.is_contiguous(memory_format=fmt):
                    fmts.append(fname)
            except Exception:
                pass
        lines.append("{}shape={} dtype={} device={} stride={} [{}]".format(
            prefix, tuple(value.shape), str(value.dtype).replace("torch.", ""),
            value.device, value.stride(), ",".join(fmts) or "non-contiguous"))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            lines.extend(describe(item, "{}[{}] ".format(prefix, i)))
    else:
        lines.append("{}{!r}".format(prefix, value))
    return lines


def run_task(task):
    import torch

    sys.path.insert(0, BASELINES)
    mod = __import__(task)

    dev = ks_device.device_str()
    torch.manual_seed(42)
    init_args = mod.get_init_inputs() or []
    model = mod.Model(*init_args)
    if hasattr(model, "to"):
        model = model.to(dev)
    if hasattr(model, "eval"):
        model.eval()

    torch.manual_seed(42)
    raw = mod.get_inputs() or []
    inputs = [t.to(dev) if isinstance(t, torch.Tensor) else t for t in raw]

    print("  inputs as the operator receives them:")
    for line in describe(inputs):
        print("    " + line)

    with torch.no_grad():
        out = model.forward(*inputs)
    ks_device.synchronize()
    print("  output:")
    for line in describe(out):
        print("    " + line)

    def call():
        with torch.no_grad():
            model.forward(*inputs)

    ms = ks_device.bench(call, warmup=25, repeat=100)
    print("  BASELINE_MS {:.4f}".format(ms))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    ks_device.require_accel()

    if args.task:
        run_task(args.task)
        return 0

    made = prepare()
    missing = [t for t in ORDER if t not in made]
    if missing:
        print("WARNING: missing task sources: {}".format(", ".join(missing)))

    print("device: {}\n".format(ks_device.device_name()))
    results = {}
    total = len([t for t in ORDER if t in made])
    for i, task in enumerate([t for t in ORDER if t in made], 1):
        print("[{}/{}] {}".format(i, total, task))
        sys.stdout.flush()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--task", task],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT after {} s".format(args.timeout))
            results[task] = None
            continue
        text = proc.stdout.decode("utf-8", "replace").rstrip()
        print(text)
        if proc.returncode != 0:
            sig = "SIGSEGV" if proc.returncode == -11 else "exit {}".format(proc.returncode)
            print("  *** {} ***".format(sig))
            results[task] = None
            continue
        found = re.search(r"BASELINE_MS ([\d.]+)", text)
        results[task] = float(found.group(1)) if found else None
        print("")
        sys.stdout.flush()

    print("\n=== baseline summary ===")
    print("{:<32} {:>10}".format("task", "v0_ms"))
    print("-" * 44)
    for task in ORDER:
        if task not in results:
            continue
        ms = results[task]
        print("{:<32} {:>10}".format(task, "{:.4f}".format(ms) if ms else "FAILED"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
