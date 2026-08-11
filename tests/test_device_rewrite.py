"""Check the device-literal rewrite that stage 3 applies to the task sources.

This is the one transformation in the harness that can go wrong quietly. Too
greedy and it corrupts the reference implementations, so every baseline is
measured against the wrong thing; too narrow and the sources that hardcode
device="cuda" fail on an NPU and those operators drop out of the run. Neither
shows up as an obvious error, so the behaviour is pinned here.

Runs without torch or an accelerator, so it can be checked anywhere.

    python3 tests/test_device_rewrite.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "tasks")


# Kept identical to stage 3's prepare(); this test exists to pin the behaviour.
def rewrite(src, accel):
    if accel == "cuda":
        return src
    src = re.sub(r'(["\'])cuda\1', r'\g<1>{}\g<1>'.format(accel), src)
    return src.replace(".cuda()", '.to("{}")'.format(accel))


CASES = (
    ('torch.randn(4, device="cuda")', 'torch.randn(4, device="npu")'),
    ("torch.randn(4, device='cuda')", "torch.randn(4, device='npu')"),
    ("model = Model().cuda().eval()", 'model = Model().to("npu").eval()'),
    # Must not touch identifiers, comments or unrelated words containing cuda.
    ("# runs on cuda hardware", "# runs on cuda hardware"),
    ("cuda_stream = 0", "cuda_stream = 0"),
    ("torch.cuda.synchronize()", "torch.cuda.synchronize()"),
)


def main():
    failures = 0

    for src, want in CASES:
        got = rewrite(src, "npu")
        if got != want:
            failures += 1
            print("FAIL {!r}".format(src))
            print("     want {!r}".format(want))
            print("     got  {!r}".format(got))
        else:
            print("ok   {!r}".format(src))

    if rewrite('device="cuda"', "cuda") != 'device="cuda"':
        print("FAIL rewrite must be a no-op on a cuda host")
        failures += 1
    else:
        print("ok   no-op on a cuda host")

    print("\n--- real task sources ---")
    if not os.path.isdir(TASKS):
        print("tasks/ not found next to this test; skipping")
        return 1 if failures else 0

    for name in sorted(os.listdir(TASKS)):
        if not name.endswith(".txt"):
            continue
        src = open(os.path.join(TASKS, name), encoding="utf-8").read()
        out = rewrite(src, "npu")
        leftover = re.findall(r'(["\'])cuda\1', out) + re.findall(r"\.cuda\(\)", out)
        n_changed = sum(1 for a, b in zip(src.splitlines(), out.splitlines()) if a != b)
        if leftover:
            print("FAIL {:<32} {} device literal(s) survived".format(name, len(leftover)))
            failures += 1
        else:
            print("ok   {:<32} {} line(s) rewritten".format(name, n_changed))

        try:
            compile(out, name, "exec")
        except SyntaxError as exc:
            print("FAIL {:<32} rewritten source no longer parses: {}".format(name, exc))
            failures += 1

    print("\n{}".format("all checks passed" if not failures
                        else "{} check(s) FAILED".format(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
