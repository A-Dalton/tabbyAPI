import platform
import torch
from common.logger import xlogger


def exllama_supports_nccl():
    if platform.system() == "Windows":
        unsupported_message = "The NCCL tensor parallel backend is not supported on Windows."
        xlogger.warning(unsupported_message)
        return False

    return torch.cuda.is_available() and torch.distributed.is_nccl_available()


def exllama_supports_cpu_kernels():
    """
    Whether ExLlamaV3's CPU-side kernels are available. The native tensor parallel
    backend's reduce and CPU MoE inference are AVX code that ExLlamaV3 only builds
    for x86; on other CPUs (ARM) they are stubbed out and fail when called.
    """

    return platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")
