"""Central JAX configuration for metacountregressor.

Single source of truth for backend/device selection, float64 precision and
GPU memory behaviour.  Every JAX-heavy module calls :func:`configure_jax`
instead of poking ``jax.config`` directly, so CPU and GPU runs behave the
same and shared-cluster GPU nodes are not starved of memory.

Environment variables (all optional)
------------------------------------
METACOUNT_JAX_PLATFORM
    ``auto`` (default) | ``cpu`` | ``gpu`` | ``tpu``.  ``auto`` never forces
    a platform: whatever JAX detects is used, so the same script picks up a
    GPU automatically when one is visible and falls back to CPU otherwise.
METACOUNT_GPU_PREALLOCATE
    ``0``/``false`` (default) disables XLA's preallocation so several jobs
    can share one GPU; ``1``/``true`` restores XLA's default behaviour of
    grabbing ~75% of VRAM up front (fastest for exclusive single-job GPUs).
JAX_PLATFORMS
    Standard JAX variable.  Honoured when METACOUNT_JAX_PLATFORM is unset.

Notes
-----
* XLA initialises its backend lazily on first use, so environment variables
  set here still take effect even after ``import jax`` has already run.
* :func:`configure_jax` is idempotent and safe to call from every module.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

__all__ = ["configure_jax", "device_summary", "get_device_info"]

_CONFIGURED = False
_INFO: Dict[str, Any] = {}


def _env_flag(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "full"}


def configure_jax(
    platform: Optional[str] = None,
    enable_x64: bool = True,
    gpu_preallocate: Optional[bool] = None,
) -> Dict[str, Any]:
    """Configure the JAX backend once per process.

    Parameters
    ----------
    platform:
        ``None`` (auto-detect), or ``'cpu'`` / ``'gpu'`` / ``'tpu'``.
        Overrides the ``METACOUNT_JAX_PLATFORM`` environment variable.
    enable_x64:
        Enable double precision (the package's estimation routines require
        it; left on unconditionally by all callers).
    gpu_preallocate:
        ``False`` lets XLA grow GPU memory on demand (safe default for
        shared nodes), ``True`` restores eager 75% preallocation.  ``None``
        defers to ``METACOUNT_GPU_PREALLOCATE`` / ``XLA_PYTHON_CLIENT_PREALLOCATE``.

    Returns
    -------
    dict
        Summary describing platform, devices and active settings.
    """
    global _CONFIGURED, _INFO
    if _CONFIGURED:
        return _INFO

    # ── Resolve requested platform BEFORE the backend initialises ────────
    requested = platform
    if requested is None:
        requested = os.environ.get("METACOUNT_JAX_PLATFORM", "").strip().lower() or None
    if requested is None:
        # Honour a standard JAX_PLATFORMS setting if the user supplied one;
        # otherwise stay fully automatic.
        requested = os.environ.get("JAX_PLATFORMS", "").strip().lower() or None
    if requested in ("", "auto", "any", "default"):
        requested = None

    if requested == "gpu":
        requested = "cuda"  # jax.config accepts backend names

    # ── GPU memory policy: grow-on-demand unless explicitly enabled ──────
    if gpu_preallocate is None:
        gpu_preallocate = _env_flag("METACOUNT_GPU_PREALLOCATE", default=False)
    if not gpu_preallocate and "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    elif gpu_preallocate:
        # Explicit opt-in removes any previously forced 'false'.
        if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "false":
            del os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]

    try:
        import jax  # noqa: WPS433 - deferred so env vars above land first
    except ImportError as _jax_exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "metacountregressor requires the 'jax' and 'jaxlib' packages.  "
            "Install the backend with:  pip install jax jaxlib jaxopt"
        ) from _jax_exc
    if requested is not None:
        try:
            jax.config.update("jax_platform_name", requested)
        except Exception:  # pragma: no cover - very old/new jax versions
            pass
    try:
        jax.config.update("jax_enable_x64", bool(enable_x64))
    except Exception:  # pragma: no cover
        pass

    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"
    try:
        devices = [str(d) for d in jax.devices()]
    except Exception:
        devices = []

    _INFO = {
        "backend": backend,
        "requested_platform": requested,
        "x64": bool(enable_x64),
        "gpu_preallocate": bool(gpu_preallocate),
        "n_devices": len(devices),
        "devices": devices,
        "jax_version": getattr(jax, "__version__", "unknown"),
    }
    _CONFIGURED = True
    return _INFO


def get_device_info() -> Dict[str, Any]:
    """Return the configuration summary recorded by :func:`configure_jax`."""
    if not _CONFIGURED:
        return configure_jax()
    return dict(_INFO)


def device_summary() -> str:
    """Human-readable one-line summary, e.g. for PBS job logs."""
    info = get_device_info()
    dev = ", ".join(info.get("devices", [])) or "none"
    return (
        f"metacountregressor JAX {info.get('jax_version', '?')} | "
        f"backend={info.get('backend', '?')} | "
        f"devices=[{dev}] | x64={info.get('x64')} | "
        f"gpu_preallocate={info.get('gpu_preallocate')}"
    )
