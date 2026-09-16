import mlx.core as mx

from ..cache import KVCache, KVCacheAllocation


class HyV4KVCache(KVCache):
    allocation_policy = KVCacheAllocation(growth_factor=2)
    memory_profile = KVCache.memory_profile

    def update_and_fetch(self, keys, values):
        previous = self.offset
        required = previous + keys.shape[2]
        capacity = 0 if self.keys is None else self.keys.shape[2]
        new_capacity = self.allocation_policy.capacity_for_update(
            capacity, previous, keys.shape[2], step=self.step
        )
        if new_capacity > capacity:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            capacity = new_capacity
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
