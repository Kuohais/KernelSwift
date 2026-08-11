"""Stage 1: what is actually installed on this machine.

Deliberately written for old Python (no f-strings with '=', no match, no PEP 604
unions) because its whole job is to still run and report when the rest cannot.
The official auto_bench.py needs Python 3.10, so if this reports 3.8 that is the
first thing to fix and nothing else in the bundle will work.

Everything is wrapped: a missing module or a driver that refuses to answer is a
finding to report, not a reason to stop.
"""

import os
import platform
import subprocess
import sys


def show(label, value):
    print("  {:<34} {}".format(label, value))


def probe(label, fn):
    try:
        show(label, fn())
    except Exception as exc:
        show(label, "UNAVAILABLE ({}: {})".format(type(exc).__name__, exc))


def run_cmd(cmd, limit=30):
    """Run a shell command, returning its first `limit` output lines."""
    try:
        out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=120)
    except Exception as exc:
        return "failed to execute: {}".format(exc)
    text = out.stdout.decode("utf-8", "replace").strip()
    if not text:
        return "(no output, exit {})".format(out.returncode)
    lines = text.splitlines()
    if len(lines) > limit:
        lines = lines[:limit] + ["... ({} more lines)".format(len(lines) - limit)]
    return "\n".join(lines)


def section(title):
    print("\n=== {} ===".format(title))


def main():
    section("interpreter")
    show("python", sys.version.replace("\n", " "))
    show("python >= 3.10 (auto_bench needs it)", sys.version_info[:2] >= (3, 10))
    show("platform", platform.platform())
    show("machine", platform.machine())

    section("torch")
    try:
        import torch
    except Exception as exc:
        show("torch", "IMPORT FAILED: {}".format(exc))
        print("\nNothing else can be measured without torch.")
        return 1
    show("torch", torch.__version__)
    probe("torch build", lambda: torch.__file__)

    section("torch_npu / Ascend backend")
    try:
        import torch_npu
        show("torch_npu", getattr(torch_npu, "__version__", "present, version unknown"))
    except Exception as exc:
        show("torch_npu", "IMPORT FAILED: {}".format(exc))

    found = []
    for name in ("npu", "cuda", "mlu", "gcu"):
        mod = getattr(torch, name, None)
        if mod is None:
            show("torch.{}".format(name), "absent")
            continue
        try:
            avail = mod.is_available()
        except Exception as exc:
            show("torch.{}".format(name), "present, is_available() raised {}".format(exc))
            continue
        show("torch.{}".format(name), "available={}".format(avail))
        if not avail:
            continue
        found.append(name)
        probe("  {} device_count".format(name), lambda m=mod: m.device_count())
        probe("  {} device_name".format(name), lambda m=mod: m.get_device_name(0))
        probe("  {} capability".format(name), lambda m=mod: m.get_device_capability(0))

        def props(m=mod):
            p = m.get_device_properties(0)
            keep = [a for a in dir(p) if not a.startswith("_")]
            return "; ".join("{}={}".format(a, getattr(p, a)) for a in sorted(keep))
        probe("  {} properties".format(name), props)

    section("triton")
    try:
        import triton
        show("triton", getattr(triton, "__version__", "present, version unknown"))
        show("triton path", getattr(triton, "__file__", "?"))
    except Exception as exc:
        show("triton", "IMPORT FAILED: {}".format(exc))
    else:
        def backends():
            import triton.backends as tb
            return sorted(getattr(tb, "backends", {}).keys())
        probe("registered backends", backends)

        def active_target():
            from triton.runtime import driver
            return str(driver.active.get_current_target())
        probe("active target", active_target)

        # CompiledKernel.run is bound per instance, not on the class, so it is
        # only inspectable after something is actually compiled. Stage 2 does
        # that; probing the class here would report a misleading AttributeError.

    for mod_name in ("triton_ascend", "bishengir", "triton.backends.ascend"):
        def imp(n=mod_name):
            m = __import__(n, fromlist=["__version__"])
            return getattr(m, "__version__", "present")
        probe("module {}".format(mod_name), imp)

    section("CANN / driver")
    show("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH", "(unset)"))
    show("ASCEND_TOOLKIT_HOME", os.environ.get("ASCEND_TOOLKIT_HOME", "(unset)"))
    show("ASCEND_OPP_PATH", os.environ.get("ASCEND_OPP_PATH", "(unset)"))
    print("\n-- npu-smi info --")
    print(run_cmd("npu-smi info", limit=40))
    print("\n-- chip and AI core count --")
    # The AI core count decides the useful grid size, and nothing in the torch
    # API reports it, so it has to come from the driver tool.
    print(run_cmd("npu-smi info -t board -i 0; npu-smi info -t common -i 0", limit=40))
    print("\n-- CANN version file --")
    print(run_cmd("cat /usr/local/Ascend/ascend-toolkit/latest/*/ascend_toolkit_install.info "
                  "2>/dev/null || cat /usr/local/Ascend/ascend-toolkit/latest/version.info 2>/dev/null"))
    print("\n-- compilers on PATH --")
    print(run_cmd("which ccec bisheng bishengir-compile ascendc gcc python3 2>/dev/null"))

    section("host")
    print(run_cmd("nproc; free -g | head -2; uname -r"))

    section("verdict")
    if found:
        show("usable accelerator", ", ".join(found))
        return 0

    show("usable accelerator", "NONE")
    print("""
No accelerator is usable from Python, so stages 2 and 3 would only repeat the
same failure ten times over. Stopping here.

On Ascend this is nearly always the CANN environment not being loaded in this
shell. Try:

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    bash run_all.sh

If torch_npu itself failed to import, its error is in the torch_npu line above
and is the thing to fix first.""")
    return 1


if __name__ == "__main__":
    sys.exit(main())
