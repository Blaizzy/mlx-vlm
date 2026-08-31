import mlx.core as mx

from ..cache import KVCache


class HyV4KVCache(KVCache):
    def update_and_fetch(self, keys, values):
        previous = self.offset
        required = previous + keys.shape[2]
        capacity = 0 if self.keys is None else self.keys.shape[2]
        if required > capacity:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            if keys.shape[2] < self.step:
                capacity = required
            else:
                capacity = max(required, 2 * max(capacity, keys.shape[2]))
            capacity = ((capacity + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((B, n_kv_heads, capacity, k_head_dim), keys.dtype)
            new_v = mx.zeros((B, n_kv_heads, capacity, v_head_dim), values.dtype)
            if self.keys is not None:
                new_k[..., :previous, :] = self.keys[..., :previous, :]
                new_v[..., :previous, :] = self.values[..., :previous, :]
            self.keys, self.values = new_k, new_v

        self.offset = required
        self.keys[..., previous : self.offset, :] = keys
        self.values[..., previous : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def extract(self, idx):
        cache = HyV4KVCache()
        if self.keys is None:
            if idx not in (0, -1):
                raise IndexError("KVCache row index out of range")
            return cache

        batch_size = int(self.keys.shape[0])
        if idx < 0:
            idx += batch_size
        if idx < 0 or idx >= batch_size:
            raise IndexError(
                f"KVCache row index {idx} out of range for batch size {batch_size}"
            )

        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, : self.offset, :])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, : self.offset, :])
        cache.offset = self.offset
        return cache
