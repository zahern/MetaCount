import jax
import jax.numpy as jnp
from jax import device_put, devices

class Device:
    def __init__(self):
        # Default device: JAX automatically picks the best available (GPU if possible)
        self._using_gpu = any("gpu" in d.device_kind.lower() for d in devices())
        self.backend = "gpu" if self._using_gpu else "cpu"

    def enable_gpu_acceleration(self, device_id=0):
        gpus = [d for d in devices() if "gpu" in d.device_kind.lower()]
        if not gpus:
            raise RuntimeError("No GPU devices available in JAX backend.")
        self.backend = "gpu"
        self._using_gpu = True
        self.device = gpus[device_id]
        print(f"Using GPU device: {self.device}")

    def disable_gpu_acceleration(self):
        self.backend = "cpu"
        self._using_gpu = False
        self.device = jax.devices("cpu")[0]
        print("Switched to CPU backend")

    @property
    def using_gpu(self):
        return self._using_gpu

    def cust_einsum(self, expr, a, b):
        """Efficient einsum with some optimized patterns"""
        if expr == 'njk,nkr -> njr':
            return jnp.matmul(a, b)
        elif expr == 'njk,k -> nj':
            return jnp.matmul(a, b)
        elif expr == 'njr,njk -> nkr':
            return jnp.matmul(jnp.swapaxes(b, 1, 2), a)
        else:
            return jnp.einsum(expr, a, b)

    def to_device(self, arr):
        if arr is None:
            return arr
        device = jax.devices(self.backend)[0]
        return device_put(arr, device)

    def to_cpu(self, arr):
        """Returns a numpy array on CPU."""
        if arr is None:
            return arr
        return jax.device_get(arr)

    def nan_safe_sum(self, arr, axis=0):
        arr = jnp.nan_to_num(arr)
        return jnp.sum(arr, axis=axis)

    def get_device_count(self):
        return len(devices())

# Example usage
device = Device()

