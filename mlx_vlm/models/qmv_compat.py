"""Keep mlx-vlm's exact quantized kernels bit-identical to MLX's own qmv.

Several Metal kernels here (speculative verify, MoE hyper-connection, fused
switch projections, GLM MTP head) re-implement MLX's affine ``qmv`` so their
output matches ``nn.QuantizedLinear`` bit for bit. That includes the input sum
used for the affine bias term (``bias * sum(x)``).

MLX 0.32.4 (ml-explore/mlx#4516) changed how ``load_vector`` forms that sum:
each group of 4 or 8 inputs is now widened to float before it is added
(``U(x[i]) + U(x[i + 1]) + ...``). Earlier versions added the group in the
input dtype (bfloat16/float16) and only then accumulated into float. Mirror
whichever behaviour the installed MLX has, so the kernels stay exact on both.
"""

import re

import mlx.core as mx


def mlx_widens_qmv_sum(version: str) -> bool:
    """True when MLX ``version`` sums qmv input groups in float (>= 0.32.4)."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return True
    return tuple(int(part) for part in match.groups()) >= (0, 32, 4)


def qmv_sum_define(version: str) -> str:
    """Metal ``#define`` for ``MLX_VLM_QMV_SUM`` matching MLX ``version``."""
    body = "static_cast<float>(v)" if mlx_widens_qmv_sum(version) else "(v)"
    return f"#ifndef MLX_VLM_QMV_SUM\n#define MLX_VLM_QMV_SUM(v) ({body})\n#endif\n"


QMV_SUM_DEFINE = qmv_sum_define(mx.__version__)
