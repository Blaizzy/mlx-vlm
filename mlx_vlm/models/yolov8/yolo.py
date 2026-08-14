import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open
from mlx.utils import tree_unflatten

class Conv(nn.Module):
    """Standard convolution with BatchNorm and SiLU activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # Ultralytics pads to keep spatial dimensions by default if p=None: p = k // 2
        pad = p if p is not None else k // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=pad, groups=g, bias=False)
        self.bn = nn.BatchNorm(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU() if act else nn.Identity()

    def __call__(self, x):
        # MLX expects inputs as (B, H, W, C)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        if self.add:
            return x + self.cv2(self.cv1(x))
        return self.cv2(self.cv1(x))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = [Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)]

    def __call__(self, x):
        y = list(mx.split(self.cv1(x), 2, axis=-1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(mx.concatenate(y, axis=-1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        
        # MaxPool2d in MLX has (B, H, W, C) layout
        pad = k // 2
        # Use simple pad and max_pool2d
        self.m_k = k
        self.m_pad = pad

    def __call__(self, x):
        x = self.cv1(x)
        # MLX padding syntax: ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0))
        pad_x = mx.pad(x, ((0,0), (self.m_pad, self.m_pad), (self.m_pad, self.m_pad), (0,0)))
        y1 = mx.max_pool2d(pad_x, kernel_size=self.m_k, stride=1)
        
        pad_y1 = mx.pad(y1, ((0,0), (self.m_pad, self.m_pad), (self.m_pad, self.m_pad), (0,0)))
        y2 = mx.max_pool2d(pad_y1, kernel_size=self.m_k, stride=1)
        
        pad_y2 = mx.pad(y2, ((0,0), (self.m_pad, self.m_pad), (self.m_pad, self.m_pad), (0,0)))
        y3 = mx.max_pool2d(pad_y2, kernel_size=self.m_k, stride=1)
        
        return self.cv2(mx.concatenate([x, y1, y2, y3], axis=-1))


def sanitize_yolo(weights_file):
    """Converts PyTorch state dict from Ultralytics into MLX format"""
    mlx_weights = {}
    with safe_open(weights_file, framework="pt") as st:
        for key in st.keys():
            tensor = st.get_tensor(key)
            
            # PyTorch conv shape: (O, I, H, W)
            # MLX conv shape: (O, H, W, I)
            if tensor.ndim == 4:
                # Transpose for Conv2d
                tensor = tensor.permute(0, 2, 3, 1)
                
            mlx_weights[key] = mx.array(tensor.numpy())
            
    return mlx_weights

if __name__ == "__main__":
    weights = sanitize_yolo('/var/folders/69/n3p05xld14g7f0tk90l220fc0000gn/T/opencode/yolo_mlx/yolov8_weights.safetensors')
    print(f"Loaded {len(weights)} keys mapped to MLX tensors.")
    
    # Test building the first few layers
    print("Testing MLX mapping of YOLO layers...")
    # model.0.conv
    conv0 = Conv(3, 64, 3, 2)
    # Update weights for conv0
    layer_weights = {
        'conv.weight': weights['model.0.conv.weight'],
        'bn.weight': weights['model.0.bn.weight'],
        'bn.bias': weights['model.0.bn.bias'],
        'bn.running_mean': weights['model.0.bn.running_mean'],
        'bn.running_var': weights['model.0.bn.running_var']
    }
    conv0.update(tree_unflatten(list(layer_weights.items())))
    
    # Dummy image (B, H, W, C)
    x = mx.random.normal((1, 640, 640, 3))
    out = conv0(x)
    print(f"Conv0 output shape (Expected 1, 320, 320, 64): {out.shape}")
    
