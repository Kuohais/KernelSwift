"""Make torch.npu exist before anything looks for it.

torch_npu registers the Ascend backend as a side effect of being imported, so
until that happens `torch.npu` is absent and every accelerator-detection loop --
including the one inside the official auto_bench.py -- concludes the machine is
CPU only. auto_bench.py is the scoring script and is kept byte-identical, so the
import has to happen outside it.

Python imports sitecustomize automatically at interpreter start for any module
on sys.path, and run_all.sh puts this directory on PYTHONPATH. Every process the
runner spawns therefore gets the backend registered without importing anything
by hand.

Silent on non-Ascend machines, so the same bundle still runs on a CUDA box.
"""

try:
    import torch  # noqa: F401
except Exception:
    pass
else:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
