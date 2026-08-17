"""One-command survey of an Ascend machine: run the stages, collect one report.

Stages run in order and each one's output is both streamed to the terminal (so
progress is visible while it works) and kept, so the single file at the end
contains everything without anyone having to scroll back.

Stage 1 is the gate. If torch or the NPU backend is missing there is no point
running the rest, and the report says so in one line rather than burying it in
tracebacks.

    bash run_all.sh                 # everything
    python3 runner.py --stage 2     # just one stage, for a re-run
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

STAGES = (
    (1, "environment", "stages/s1_env.py", 900,
     "what torch / torch_npu / triton / CANN are installed"),
    (2, "triton", "stages/s2_triton.py", 5400,
     "whether Triton works here, what a launch costs, which constructs break"),
    (3, "baselines", "stages/s3_baselines.py", 5400,
     "reference latency and input layout for all ten operators"),
    (5, "launch", "stages/s5_launch.py", 1800,
     "what a launch costs and whether consecutive launches overlap"),
)


def stream(cmd, log_path):
    """Run cmd, echoing output live and returning (returncode, text)."""
    chunks = []
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1,
                                universal_newlines=True, cwd=ROOT)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            chunks.append(line)
        proc.wait()
    return proc.returncode, "".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, action="append",
                        help="run only these stages (repeatable)")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    wanted = args.stage or [n for n, _, _, _, _ in STAGES]

    report_path = os.path.join(RESULTS, "REPORT.txt")

    def flush_report(header):
        """Rebuild the report from whatever stage logs exist on disk.

        Assembling from the logs rather than from this run's output is what
        makes `--stage 5` safe: re-running one stage refreshes its log and
        leaves the others in place, so the report stays complete instead of
        shrinking to just the stage that was re-run. It also means the report is
        current after every stage, so interrupting a long run still leaves a
        file worth sending.
        """
        parts = [header]
        for num, name, _script, _timeout, blurb in STAGES:
            log_path = os.path.join(RESULTS, "stage{}_{}.log".format(num, name))
            if not os.path.exists(log_path):
                continue
            parts.append("\n{0}\n### stage {1}: {2} -- {3}\n{0}".format(
                "=" * 78, num, name, blurb))
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                parts.append(fh.read().rstrip())
        with open(report_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(parts) + "\n")

    started = time.time()
    header = "KernelSwift Ascend survey, written {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S"))
    print(header)

    for num, name, script, timeout, blurb in STAGES:
        if num not in wanted:
            continue
        banner = "\n{0}\n### stage {1}: {2} -- {3}\n{0}".format(
            "=" * 78, num, name, blurb)
        print(banner)
        sys.stdout.flush()

        log_path = os.path.join(RESULTS, "stage{}_{}.log".format(num, name))
        t0 = time.time()
        try:
            code, _text = stream([sys.executable, script], log_path)
        except Exception as exc:
            code = 1
            print("runner failed to start stage: {}".format(exc))
        elapsed = "\n[stage {} finished in {:.0f} s, exit {}]".format(
            num, time.time() - t0, code)
        print(elapsed)
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(elapsed + "\n")
        flush_report(header)
        sys.stdout.flush()

        if num == 1 and code != 0:
            print("\nStage 1 failed, so torch or the accelerator is not usable "
                  "on this machine. Stopping: later stages would only produce "
                  "noise. Send back this report as it is.")
            break

    print("\n\nTotal {:.0f} s".format(time.time() - started))
    flush_report(header)

    archive = os.path.join(ROOT, "ks-ascend-results.tar.gz")
    subprocess.run(["tar", "-czf", archive, "-C", ROOT, "results"], cwd=ROOT)

    print("\n" + "=" * 78)
    print("Report:  {}".format(report_path))
    print("Archive: {}".format(archive))
    print("Send back either one. The archive also holds the per-stage logs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
