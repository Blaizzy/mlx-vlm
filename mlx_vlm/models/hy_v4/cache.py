from dataclasses import replace

import mlx.core as mx

from ..cache import KVCache


class HyV4KVCache(KVCache):
    def memory_profile(self, token_count):
        profile = super().memory_profile(token_count)
        return replace(profile, bytes_per_token=2 * profile.bytes_per_token)

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
        cache = super().extract(idx)
        return HyV4KVCache.from_state(cache.state, cache.meta_state)
