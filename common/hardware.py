import importlib.util
import platform
from typing import Optional

from common.logger import xlogger

# Oldest NVIDIA compute capability ExLlamaV3's kernels run on: Turing (20 series).
# Volta is 7.0 and lacks instructions they rely on
MIN_CUDA_COMPUTE_CAPABILITY = (7, 5)


def _torch():
    """
    torch, imported on first use. main.py asks this module about the installed
    build before it sets the allocator environment, and torch reads that
    environment when it is imported, so importing it here at module level would
    pin the allocator to its default before the config has been applied.
    """

    import torch

    return torch


def installed_torch_is_rocm() -> bool:
    """
    Whether the installed torch package is a ROCm build, read from its version
    metadata so it can be known before torch is imported (some environment
    variables only take effect if set before the import).
    """

    try:
        from importlib.metadata import version

        return "rocm" in version("torch")
    except Exception:
        return False


def torch_backend() -> Optional[str]:
    """The GPU backend the installed PyTorch was built for: "cuda", "rocm" or None."""

    torch = _torch()
    if torch.version.hip:
        return "rocm"
    if torch.version.cuda:
        return "cuda"
    return None


def torch_gpu_problem() -> Optional[str]:
    """
    A message explaining why the installed PyTorch can't run models on a GPU, or
    None if it can. Catches the two setups that otherwise fail in confusing
    ways: a CPU-only PyTorch build (typically from a plain `pip install torch`
    replacing the CUDA or ROCm one), and a GPU build that finds no device.
    """

    torch = _torch()
    backend = torch_backend()
    if backend is None:
        return (
            f"The installed PyTorch ({torch.__version__}) is a CPU-only build, with neither "
            "CUDA nor ROCm support, so ExLlamaV3 cannot use a GPU. This usually happens when "
            "torch was installed or upgraded with a plain `pip install torch`, replacing the "
            "GPU build. Reinstall the dependencies with the update script in update_scripts/ "
            "or `pip install --upgrade .[cu12]` (or `.[cu13]`), or install a ROCm build of "
            "PyTorch for AMD GPUs."
        )

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        if backend == "rocm":
            built_for = f"ROCm (HIP {torch.version.hip})"
            hint = "an AMD GPU with a working ROCm installation"
        else:
            built_for = f"CUDA {torch.version.cuda}"
            hint = "an NVIDIA GPU with a current driver"
        return (
            f"The installed PyTorch ({torch.__version__}) is built for {built_for} but found "
            f"no usable GPU. Check that {hint} is present, and that CUDA_VISIBLE_DEVICES or "
            "HIP_VISIBLE_DEVICES, if set, don't hide it."
        )

    return None


def announce_exllamav3_source_build():
    """
    Say so before ExLlamaV3 is imported when it has no prebuilt extension, as on
    ARM where the pyproject installs its universal wheel. The import then compiles
    the extension, which takes a while on the first start (later starts reuse the
    cached build) and needs nvcc, so a missing toolkit is reported up front.
    """

    if importlib.util.find_spec("exllamav3_ext") is not None:
        return

    xlogger.info(
        f"ExLlamaV3 has no prebuilt extension for this platform ({platform.machine()}), "
        "so it is compiled on first use and cached. The first start can take a while."
    )

    torch = _torch()
    if torch_backend() != "cuda":
        return

    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        xlogger.error(
            f"Compiling ExLlamaV3 needs nvcc from a CUDA {torch.version.cuda.split('.')[0]}.x "
            "toolkit, which was not found. Install the toolkit and add its bin folder to PATH "
            "or set CUDA_HOME. See the ARM page in the wiki."
        )


def hardware_supports_exllamav3(gpu_device_list: list[int]):
    """
    Check whether all GPUs in the list can run ExLlamaV3.

    On CUDA builds every device needs compute capability 7.5 (Turing) or higher.
    ROCm builds skip the check: the capability number is NVIDIA's and means
    nothing for AMD devices, and ExLlamaV3 does its own validation there.
    """

    if torch_backend() == "rocm":
        return True

    if not gpu_device_list:
        return False

    torch = _torch()
    min_compute_capability = min(
        torch.cuda.get_device_capability(device=device_idx) for device_idx in gpu_device_list
    )

    return min_compute_capability >= MIN_CUDA_COMPUTE_CAPABILITY
