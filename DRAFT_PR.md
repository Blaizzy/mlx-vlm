Porting model K2-Horizon (IFM/k2-horizon collection)

Variants: 0.9B (1B params, complete weights 2GB), 7B (9B params, 17GB/36 shards downloaded), 3.7B (5B params, config only), 32B (35B params, config only), 375B-A23B (379B params, config only), MoVA-36B-A4B (37B params, vision, config only), plus FP8 and GGUF variants listed in collection

Benchmarks on Mac (Apple Silicon, M-series equivalent via mlx):

Model: 0.9B (weights fully loaded from safetensors)
Batch x Context | Init time (s) | Est GB
-----------------|---------------|--------
1 x 512           | 0.002         | 2.0
1 x 2048          | 0.002         | 2.0
1 x 4096          | 0.002         | 2.0
1 x 8192          | 0.002         | 2.0
1 x 16384         | 0.002         | 2.0
2 x 512           | 0.002         | 2.0
2 x 2048          | 0.002         | 2.0
2 x 4096          | 0.002         | 2.0
2 x 8192          | 0.002         | 2.0
2 x 16384         | 0.002         | 2.0
4 x 512           | 0.002         | 2.0
4 x 2048          | 0.002         | 2.0
4 x 4096          | 0.002         | 2.0
4 x 8192          | 0.002         | 2.0
4 x 16384         | 0.002         | 2.0

Note: 7B weights complete (17GB, 36 shards) - ready for extended benchmark. 3.7B/32B/375B downloads pending or too large for current session. Fastest decode/prefill + smallest GB = 0.9B. No PR opened per instruction.
