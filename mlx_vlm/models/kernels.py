import mlx.core as mx


def nearest_interpolate(x, size=None, scale_factor=None):
    """
    Nearest neighbor interpolation that exactly matches PyTorch's behavior.
    """
    # Get input dimensions
    batch_size, channels, in_h, in_w = x.shape

    # Calculate output dimensions
    if size is not None:
        out_h, out_w = size
    elif scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_h = scale_w = scale_factor
        else:
            scale_h, scale_w = scale_factor
        out_h, out_w = int(in_h * scale_h), int(in_w * scale_w)
    else:
        raise ValueError("Either size or scale_factor must be specified")

    # Create dimensions tensor
    dims = mx.array([batch_size, channels, in_h, in_w, out_h, out_w], dtype=mx.int32)

    # Reshape input tensor to 1D for kernel processing
    x_flat = x.reshape(-1)
    input_dtype = x.dtype
    if input_dtype != mx.float32:
        x_flat = x_flat.astype(mx.float32)

    # Metal kernel source that matches PyTorch's coordinate calculation
    source = """
        uint x_out = thread_position_in_grid.x;
        uint y_out = thread_position_in_grid.y;
        uint bc_idx = thread_position_in_grid.z;

        int batch_size = dims[0];
        int channels = dims[1];
        int in_h = dims[2];
        int in_w = dims[3];
        int out_h = dims[4];
        int out_w = dims[5];

        if (x_out >= (uint)out_w || y_out >= (uint)out_h || bc_idx >= (uint)(batch_size * channels))
            return;

        int c = bc_idx % channels;
        int b = bc_idx / channels;

        // PyTorch's coordinate calculation for nearest neighbor
        // This matches: torch.nn.functional.interpolate(..., mode='nearest')
        float scale_h = float(in_h) / float(out_h);
        float scale_w = float(in_w) / float(out_w);

        // PyTorch uses floor for nearest neighbor coordinate mapping
        int y_in = int(floor(float(y_out) * scale_h));
        int x_in = int(floor(float(x_out) * scale_w));

        // Clamp to bounds
        y_in = max(0, min(y_in, in_h - 1));
        x_in = max(0, min(x_in, in_w - 1));

        int input_offset = ((b * channels + c) * in_h + y_in) * in_w + x_in;
        int output_offset = ((b * channels + c) * out_h + y_out) * out_w + x_out;

        output[output_offset] = input[input_offset];
    """

    # Create and run kernel
    kernel = mx.fast.metal_kernel(
        name="nearest_interpolation",
        input_names=["input", "dims"],
        output_names=["output"],
        source=source,
    )

    threadgroup = get_optimal_threadgroup(out_w, out_h)
    outputs = kernel(
        inputs=[x_flat, dims],
        grid=(out_w, out_h, batch_size * channels),
        threadgroup=threadgroup,
        output_shapes=[(batch_size * channels * out_h * out_w,)],
        output_dtypes=[mx.float32],
    )

    result = outputs[0].reshape(batch_size, channels, out_h, out_w)
    if input_dtype != mx.float32:
        result = result.astype(input_dtype)

    return result


def bicubic_interpolate(x, size=None, scale_factor=None, align_corners=False):
    """
    Bicubic interpolation using MLX's built-in interpolate function.

    Args:
        x: MLX tensor of shape [B, C, H, W]
        size: Tuple of (out_h, out_w) or None
        scale_factor: Float or tuple of (scale_h, scale_w) or None
        align_corners: Whether to align corners

    Returns:
        Interpolated MLX tensor
    """
    # Get input dimensions
    batch_size, channels, in_h, in_w = x.shape

    # Calculate output dimensions
    if size is not None:
        out_h, out_w = size
        scale_h, scale_w = out_h / in_h, out_w / in_w
    elif scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_h = scale_w = scale_factor
        else:
            scale_h, scale_w = scale_factor
        out_h, out_w = int(in_h * scale_h), int(in_w * scale_w)
    else:
        raise ValueError("Either size or scale_factor must be specified")

    # Create scale and align_corners parameters tensor
    params = mx.array(
        [scale_h, scale_w, 1.0 if align_corners else 0.0], dtype=mx.float32
    )

    # Create dimensions tensor
    dims = mx.array([batch_size, channels, in_h, in_w, out_h, out_w], dtype=mx.int32)

    # Reshape input tensor to 1D for kernel processing
    x_flat = x.reshape(-1)

    # Convert to float32 for processing if needed
    input_dtype = x.dtype
    if input_dtype != mx.float32:
        x_flat = x_flat.astype(mx.float32)

    # Metal kernel source code
    source = """
        // Get thread position
        uint x_out = thread_position_in_grid.x;
        uint y_out = thread_position_in_grid.y;
        uint bc_idx = thread_position_in_grid.z;

        // Extract dimensions from dims
        int batch_size = dims[0];
        int channels = dims[1];
        int in_h = dims[2];
        int in_w = dims[3];
        int out_h = dims[4];
        int out_w = dims[5];

        // Extract scales and flags
        float scale_h = params[0];
        float scale_w = params[1];
        bool align_corners = params[2] > 0.5;

        // Check bounds
        if (x_out >= (uint)out_w || y_out >= (uint)out_h || bc_idx >= (uint)(batch_size * channels))
            return;

        // Calculate batch and channel indices
        int c = bc_idx % channels;
        int b = bc_idx / channels;

        // Calculate input coordinates based on output position
        float x_in, y_in;

        if (align_corners && out_w > 1 && out_h > 1) {
            x_in = float(x_out) * (in_w - 1) / (out_w - 1);
            y_in = float(y_out) * (in_h - 1) / (out_h - 1);
        } else {
            // Fix the alignment calculation to ensure consistent mapping across thread boundaries
            x_in = ((float(x_out) + 0.5f) / float(out_w)) * float(in_w) - 0.5f;
            y_in = ((float(y_out) + 0.5f) / float(out_h)) * float(in_h) - 0.5f;
        }

        // Get integer and fractional parts
        int x0 = int(floor(x_in));
        int y0 = int(floor(y_in));
        float x_frac = x_in - x0;
        float y_frac = y_in - y0;

        // Improved cubic kernel function for better continuity
        auto cubic_kernel = [](float x) -> float {
            float absx = fabs(x);
            float absx2 = absx * absx;
            float absx3 = absx2 * absx;

            // Use a=-0.5 for smoother interpolation
            const float a = -0.5f;

            if (absx <= 1.0f) {
                return (a+2.0f)*absx3 - (a+3.0f)*absx2 + 1.0f;
            } else if (absx < 2.0f) {
                return a*absx3 - 5.0f*a*absx2 + 8.0f*a*absx - 4.0f*a;
            }
            return 0.0f;
        };

        // Perform bicubic interpolation with improved boundary handling
        float result = 0.0f;
        float weight_sum = 0.0f;  // Track weight sum for normalization

        for (int i = -1; i <= 2; i++) {
            int y_pos = y0 + i;
            // Clamp y coordinate to valid range
            y_pos = max(0, min(y_pos, in_h - 1));
            float wy = cubic_kernel(y_frac - i);

            for (int j = -1; j <= 2; j++) {
                int x_pos = x0 + j;
                // Clamp x coordinate to valid range
                x_pos = max(0, min(x_pos, in_w - 1));
                float wx = cubic_kernel(x_frac - j);
                float weight = wy * wx;

                // Calculate input tensor offset
                int input_offset = ((b * channels + c) * in_h + y_pos) * in_w + x_pos;

                // Add weighted contribution
                result += input[input_offset] * weight;
                weight_sum += weight;
            }
        }

        // Normalize by weight sum to ensure consistent intensity
        if (weight_sum > 0.0f) {
            result /= weight_sum;
        }

        // Calculate output tensor offset
        int output_offset = ((b * channels + c) * out_h + y_out) * out_w + x_out;

        // Assign the result to output
        output[output_offset] = (float)result;
    """

    # Create the kernel
    kernel = mx.fast.metal_kernel(
        name="bicubic_interpolation",
        input_names=["input", "dims", "params"],
        output_names=["output"],
        source=source,
    )

    # Run the kernel
    threadgroup = get_optimal_threadgroup(out_w, out_h)
    outputs = kernel(
        inputs=[x_flat, dims, params],
        grid=(out_w, out_h, batch_size * channels),
        threadgroup=threadgroup,
        output_shapes=[(batch_size * channels * out_h * out_w,)],
        output_dtypes=[mx.float32],  # Always use float32 for kernel output
    )

    # Reshape output back to 4D tensor and convert back to original dtype
    result = outputs[0].reshape(batch_size, channels, out_h, out_w)
    if input_dtype != mx.float32:
        result = result.astype(input_dtype)

    return result


def get_optimal_threadgroup(out_w, out_h):
    # Calculate optimal threadgroup dimensions based on output dimensions

    # Maximum threadgroup size for most Metal GPUs
    # This could be made more dynamic with Metal API queries if needed
    MAX_THREADS_PER_GROUP = 1024
    MAX_THREADS_PER_DIM = 1024

    # Start with a reasonable default size for 2D workloads
    default_threadgroup = (32, 32, 1)

    try:
        # Don't create threadgroups larger than the work dimensions
        max_width = min(MAX_THREADS_PER_DIM, out_w)
        max_height = min(MAX_THREADS_PER_DIM, out_h)

        # Find largest power of 2 that fits within our dimensions
        width = 2 ** (max_width.bit_length() - 1)
        if width > max_width:
            width = width // 2

        height = 2 ** (max_height.bit_length() - 1)
        if height > max_height:
            height = height // 2

        # Ensure we don't exceed maximum threads per threadgroup
        while width * height > MAX_THREADS_PER_GROUP:
            # Reduce the larger dimension first
            if width >= height:
                width = width // 2
            else:
                height = height // 2

        # Ensure minimum size for efficiency
        width = max(8, width)
        height = max(8, height)

        return (width, height, 1)

    except Exception:
        # Return safe defaults if calculation fails
        return default_threadgroup
