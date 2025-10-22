# ---- place in a utils/system.py (or a notebook cell) and call it right after loading `config`
import os, sys, math
import torch
import numexpr as ne


def apply_system_config(config):
    syscfg = config.system
    # numexpr set threads
    ne.set_num_threads(config.system.numexpr_threads)
    print("numexpr threads:", ne.get_num_threads())

    # ----- CPU / BLAS threading
    ncpu = os.cpu_count() or 4
    intra = syscfg.cpu.intra_op_threads or max(1, ncpu // 2)
    inter = syscfg.cpu.inter_op_threads or max(1, min(4, ncpu // 4))
    os.environ.setdefault("OMP_NUM_THREADS", str(syscfg.cpu.omp_num_threads or intra))
    os.environ.setdefault("MKL_NUM_THREADS", str(syscfg.cpu.mkl_num_threads or intra))
    if syscfg.cpu.kmp_affinity:
        os.environ.setdefault("KMP_AFFINITY", syscfg.cpu.kmp_affinity)
    try:
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        torch.set_num_interop_threads(inter)
    except Exception:
        pass

    # Safer multiprocessing method on Linux/macOS (no-op on Windows)
    try:
        import torch.multiprocessing as mp
        mp.set_start_method(syscfg.cpu.start_method, force=True)
    except Exception:
        pass

    # Optional process CPU affinity
    if hasattr(os, "sched_setaffinity") and syscfg.cpu.pin_affinity_cores:
        os.sched_setaffinity(0, syscfg.cpu.pin_affinity_cores)

    # ----- CUDA backend flags
    use_cuda = syscfg.cuda.enabled and torch.cuda.is_available()
    if use_cuda:
        torch.set_float32_matmul_precision(syscfg.cuda.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = bool(syscfg.cuda.allow_tf32)
        torch.backends.cudnn.benchmark = bool(syscfg.cuda.cudnn_benchmark)
        if syscfg.cuda.cudnn_deterministic is not None:
            torch.backends.cudnn.deterministic = bool(syscfg.cuda.cudnn_deterministic)

    # Return a small runtime object for convenience
    device_ids = syscfg.cuda.device_ids
    if use_cuda and (not device_ids):
        device_ids = list(range(torch.cuda.device_count()))
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_dtype = torch.bfloat16 if syscfg.cuda.amp_dtype == "bf16" else torch.float16
    scaler_enabled = syscfg.cuda.grad_scaler_enabled and syscfg.cuda.amp and (amp_dtype is torch.float16)

    return {
        "device": device,
        "device_ids": device_ids or [],
        "amp_enabled": use_cuda and syscfg.cuda.amp,
        "amp_dtype": amp_dtype,
        "scaler": torch.cuda.amp.GradScaler(enabled=scaler_enabled),
        "use_uva": syscfg.cuda.uva
    }
