"""Prototype: reuse packed NVFP4 weights within consecutive sorted routes."""

from functools import lru_cache

import mlx.core as mx

from .moe_kernel import _HEADER


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="ml4_grouped_native_nvfp4_qmv",
        input_names=["x", "w", "s", "g", "indices"],
        output_names=["out"],
        header=_HEADER + _GROUPED_HEADER,
        compile_options={"math_mode": "safe"},
        source=r"""
        const uint route0=threadgroup_position_in_grid.z;
        const uint lane=thread_index_in_simdgroup;
        const uint row0=threadgroup_position_in_grid.y*8+simdgroup_index_in_threadgroup*ROWS;
        int expert=int(indices[route0]);
        if (expert<0) {
            if (lane==0) for (uint r=0; r<ROWS; ++r) out[route0*w_shape[1]+row0+r]=T(0);
            return;
        }
        uint first=route0;
        while (first>0 && int(indices[first-1])==expert) --first;
        if ((route0-first)%V != 0) return;
        uint count=1;
        while (count<V && route0+count<ROUTES && int(indices[route0+count])==expert) ++count;
        const device T* xp=x+route0*x_shape[x_ndim-1];
        const device uint* wp=w+size_t(expert)*w_shape[1]*(x_shape[x_ndim-1]/8);
        const device uchar* sp=s+size_t(expert)*w_shape[1]*(x_shape[x_ndim-1]/16);
        if (count==1) {
            fp_qmv_fast_impl<T,16,4,true,false,ROWS>(wp,sp,g+expert,xp,out+route0*w_shape[1],
                x_shape[x_ndim-1],w_shape[1],
                uint3(0,threadgroup_position_in_grid.y*(4/ROWS),0),
                simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
            return;
        }
        grouped_qmv_impl<T,V,ROWS>(wp,sp,g+expert,xp,out+route0*w_shape[1],
            x_shape[x_ndim-1],w_shape[1],count,threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
        """,
    )


def grouped_qmv(x, linear, indices, vectors=2, rows=4):
    """Inputs contain one vector per sorted route; unowned routes use -1."""
    k, n = linear.input_dims, linear.output_dims
    e = linear.num_experts
    assert x.size // k == indices.size and k % 512 == 0 and n % 8 == 0
    if hasattr(linear, "global_scale"):
        scale = linear.global_scale.astype(mx.float32)
        factor = (
            2688.0 / scale
            if linear._scale_name == "weight_global_scale"
            else 2688.0 * scale
        )
        factor = mx.broadcast_to(factor.reshape(-1), (e,))
    else:
        factor = mx.full((e,), 2688.0, mx.float32)
    if e < 64:
        factor = mx.concatenate([factor, mx.zeros((64 - e,), mx.float32)])
    return _kernel()(
        inputs=[x, linear.weight, linear.scales, factor, indices],
        template=[
            ("T", x.dtype),
            ("V", vectors),
            ("ROUTES", indices.size),
            ("ROWS", rows),
        ],
        grid=(32, (n // 8) * (8 // rows), indices.size),
        threadgroup=(32, 8 // rows, 1),
        output_shapes=[(*indices.shape, n)],
        output_dtypes=[x.dtype],
    )[0]


_GROUPED_HEADER = r"""
template <typename T, int V, int ROWS>
METAL_FUNC void grouped_qmv_impl(
    const device uint32_t* w, const device uint8_t* scales,
    const device float* global_scale, const device T* x, device T* y,
    const constant int& in_vec_size, const constant int& out_vec_size,
    uint count, uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int values_per_thread=16;
  constexpr int block_size=512;
  constexpr int results_per_simdgroup=ROWS;
  const device uint8_t* ws=(const device uint8_t*)w;
  typedef float U;
  thread U x_thread[V][values_per_thread];
  thread U result[V][results_per_simdgroup];
  for (int v=0; v<V; ++v)
    for (int row=0; row<results_per_simdgroup; ++row) result[v][row]=0;
  const int in_vec_size_w=in_vec_size/2;
  const int in_vec_size_g=in_vec_size/16;
  const int out_row=tid.y*8+simd_gid*ROWS;
  ws+=out_row*in_vec_size_w+simd_lid*8;
  scales+=out_row*in_vec_size_g+simd_lid;
  x+=simd_lid*values_per_thread;
  y+=out_row;
  for (int k=0; k<in_vec_size; k+=block_size) {
    for (int v=0; v<V; ++v)
      load_vector<T,U,values_per_thread>(x+min(uint(v),count-1)*in_vec_size,x_thread[v]);
    for (int row=0; row<results_per_simdgroup; ++row) {
      auto wl=(const device uint8_t*)(ws+row*in_vec_size_w);
      const device auto* sl=scales+row*in_vec_size_g;
      U s=dequantize_scale<U,16>(sl[0]);
      for (int v=0; v<V; ++v)
        result[v][row]+=qdot<U,values_per_thread,4>(wl,x_thread[v],s);
    }
    ws+=block_size/2;
    scales+=block_size/16;
    x+=block_size;
  }
  float inv_scale_enc=*global_scale/(F8E4M3_MAX*F4E2M1_MAX);
  for (int v=0; v<V; ++v) {
    for (int row=0; row<results_per_simdgroup; ++row) {
      result[v][row]=simd_sum(result[v][row]);
      if (simd_lid==0 && v<count)
        y[v*out_vec_size+row]=static_cast<T>(result[v][row]*inv_scale_enc);
    }
  }
}
"""
