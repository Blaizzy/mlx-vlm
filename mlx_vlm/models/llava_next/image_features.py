import mlx.core as mx
from transformers.image_processing_utils import select_best_resolution


def pack_image_features(
    features, image_sizes, image_size, patch_size, pinpoints, newline
):
    packed = []
    side = image_size // patch_size
    for feature, (original_height, original_width) in zip(features, image_sizes):
        base = feature[0]
        if feature.shape[0] == 1:
            packed.append(mx.concatenate([base, newline[None]], axis=0))
            continue
        best_height, best_width = select_best_resolution(
            (original_height, original_width), pinpoints
        )
        grid_height, grid_width = best_height // image_size, best_width // image_size
        spatial = feature[1:].reshape(grid_height, grid_width, side, side, -1)
        spatial = spatial.transpose(0, 2, 1, 3, 4).reshape(
            grid_height * side, grid_width * side, -1
        )
        height, width = spatial.shape[:2]
        if original_width / original_height > width / height:
            retained = int(round(original_height * width / original_width, 7))
            padding = (height - retained) // 2
            spatial = spatial[padding : height - padding]
        else:
            retained = int(round(original_width * height / original_height, 7))
            padding = (width - retained) // 2
            spatial = spatial[:, padding : width - padding]
        spatial = mx.concatenate(
            [
                spatial,
                mx.broadcast_to(newline, (spatial.shape[0], 1, newline.shape[0])),
            ],
            axis=1,
        )
        packed.append(
            mx.concatenate([base, spatial.reshape(-1, newline.shape[0])], axis=0)
        )
    return packed
