"""Image-to-3D Gaussian splats, with experimental multi-image conditioning."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from ...utils import get_model_path, load_image
from .camera import decode_camera
from .decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder
from .dit import LatentSeqMMFlowModel
from .gaussian import Gaussian
from .preprocess import CROP_MARGIN, has_real_alpha, preprocess_image
from .sampler import sample_flow, sample_multiview, validate_sampling


@dataclass
class IdolOutput:
    gaussian: Gaussian
    preprocessed_image: Image.Image
    camera: dict
    path: Path | None = None
    preprocessed_images: list[Image.Image] = field(default_factory=list)
    cameras: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.preprocessed_images:
            self.preprocessed_images = [self.preprocessed_image]
        if not self.cameras:
            self.cameras = [self.camera]

    def save(self, path):
        """Save .ply, the prepared input, and predicted camera beside it."""
        path = Path(path).expanduser().with_suffix(".ply")
        self.gaussian.save_ply(path)
        self.preprocessed_image.save(path.with_name(path.stem + "_input.png"))
        path.with_name(path.stem + "_camera.json").write_text(
            json.dumps(self.camera, indent=2) + "\n"
        )
        if len(self.preprocessed_images) > 1:
            for index, image in enumerate(self.preprocessed_images[1:], start=1):
                image.save(path.with_name(f"{path.stem}_input_{index}.png"))
            path.with_name(path.stem + "_cameras.json").write_text(
                json.dumps(self.cameras, indent=2) + "\n"
            )
        self.path = path
        return path


class IdolPipeline:
    def __init__(
        self,
        weights_dir,
        *,
        dinov3=None,
        flux2_vae=None,
        dinov3_dtype="bf16",
        background_removal=True,
        local_files_only=False,
    ):
        self.weights_dir = Path(weights_dir)
        self.config = json.loads((self.weights_dir / "config.json").read_text())
        self.local_files_only = local_files_only
        self.background_removal = background_removal
        self.background_remover = None
        self.image_encoder = None
        self._dinov3 = dinov3 or self.config["image_encoders"]["dinov3"]
        self._flux2_vae = flux2_vae or self.config["image_encoders"]["flux2_vae"]
        if dinov3_dtype not in ("bf16", "fp32", "fp16"):
            raise ValueError("dinov3_dtype must be bf16, fp32, or fp16")
        self._dinov3_dtype = {
            "bf16": mx.bfloat16,
            "fp32": mx.float32,
            "fp16": mx.float16,
        }[dinov3_dtype]
        classes = {
            "denoiser": LatentSeqMMFlowModel,
            "decoder": OctreeProbabilityFixedlenDecoder,
            "decoder_gs": GaussianFixedlenDecoder,
        }
        for name, cls in classes.items():
            model = cls(**self.config[name]["args"])
            weights = model.sanitize(
                mx.load(str(self.weights_dir / f"{name}.safetensors"))
            )
            model.load_weights(list(weights.items()))
            model.eval()
            mx.eval(model.parameters())
            setattr(self, name, model)

    @classmethod
    def from_pretrained(
        cls,
        repo_or_dir="speridlabs/idol",
        *,
        revision=None,
        force_download=False,
        **kwargs,
    ):
        return cls(
            get_model_path(
                str(Path(repo_or_dir).expanduser()),
                allow_patterns=["config.json", "*.safetensors"],
                local_files_only=kwargs.get("local_files_only", False),
                revision=revision,
                force_download=force_download,
            ),
            **kwargs,
        )

    def preprocess(self, image, mode="synthetic", crop_margin=None):
        if mode not in CROP_MARGIN:
            raise ValueError("mode must be synthetic or real")
        image = load_image(image, mode="RGBA")
        if (
            not has_real_alpha(image)
            and self.background_removal
            and self.background_remover is None
        ):
            from .birefnet import BiRefNet

            # Keep the one-time background pass in float32 for numerical stability.
            model = BiRefNet()
            weights = model.sanitize(
                mx.load(str(self.weights_dir / "birefnet.safetensors"))
            )
            model.load_weights(list(weights.items()))
            model.eval()
            mx.eval(model.parameters())
            self.background_remover = model
        return preprocess_image(
            image,
            self.background_remover,
            image_size=self.config["image_size"],
            crop_margin=crop_margin if crop_margin is not None else CROP_MARGIN[mode],
        )

    def encode_image(self, image, *, seed=None):
        if self.image_encoder is None:
            from .conditioning import ImageEncoder

            self.image_encoder = ImageEncoder(
                self._dinov3,
                self._flux2_vae,
                dinov3_dtype=self._dinov3_dtype,
                local_files_only=self.local_files_only,
            )
        # Lazy module initialization also consumes RNG. Seed only after loading,
        # so the first request and subsequent requests produce the same result.
        if seed is not None:
            mx.random.seed(seed)
        pixels = mx.array(np.asarray(image, dtype=np.float32) / 255)[None]
        return self.image_encoder(pixels)

    def sample(
        self,
        condition,
        *,
        seed=42,
        steps=50,
        cfg_strength=3.0,
        rescale_t=3.0,
        verbose=False,
        batch_cfg=False,
    ):
        """Sample from a condition dict or a list of single-view condition dicts."""
        if isinstance(condition, (list, tuple)):
            if not condition:
                raise ValueError("At least one image condition is required")
            if len(condition) == 1:
                condition = condition[0]
        multiview = isinstance(condition, (list, tuple))
        if seed is not None:
            mx.random.seed(seed)
        batch = 1 if multiview else condition["feature1"].shape[0]
        noise = {
            "latent": mx.random.normal(
                (batch, self.denoiser.q_token_length, self.denoiser.in_channels)
            ),
            "camera": mx.random.normal((batch, 1, self.denoiser.cam_channels)),
        }
        sampler = sample_multiview if multiview else sample_flow
        return sampler(
            self.denoiser,
            noise,
            condition,
            steps=steps,
            cfg_strength=cfg_strength,
            rescale_t=rescale_t,
            verbose=verbose,
            batch_cfg=batch_cfg,
        )

    def _validate_num_gaussians(self, num_gaussians):
        if (
            not isinstance(num_gaussians, int)
            or not 32768 <= num_gaussians <= 1048576
            or num_gaussians % self.decoder_gs.rep_config["num_gaussians"]
        ):
            raise ValueError(
                "num_gaussians must be a multiple of 32 between 32768 and 1048576"
            )

    def decode(self, latent, num_gaussians=262144):
        if latent.shape[0] != 1:
            raise ValueError("decode expects a single object (batch size 1)")
        self._validate_num_gaussians(num_gaussians)
        points = OctreeProbabilityFixedlenDecoder.sample(
            self.decoder,
            latent,
            num_gaussians // self.decoder_gs.rep_config["num_gaussians"],
            self.config["max_voxel_level"],
        )
        features = self.decoder_gs(points, latent)
        gaussian = self.decoder_gs.to_representation(points, features)[0]
        mx.eval(
            gaussian.xyz,
            gaussian.features_dc,
            gaussian.scaling,
            gaussian.rotation,
            gaussian.opacity,
        )
        return gaussian

    def __call__(
        self,
        image,
        mode="synthetic",
        seed=42,
        steps=50,
        cfg_strength=3.0,
        rescale_t=3.0,
        num_gaussians=262144,
        crop_margin=None,
        verbose=False,
        batch_cfg=False,
    ):
        validate_sampling(steps, rescale_t, cfg_strength)
        self._validate_num_gaussians(num_gaussians)
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if not images:
            raise ValueError("At least one image is required")
        prepared = [self.preprocess(item, mode, crop_margin) for item in images]
        conditions = [None] * len(images)
        # Encode the primary view last so seeded sampling and decoding continue
        # the same RNG stream as a run with only that image, regardless of count.
        for index in reversed(range(len(images))):
            conditions[index] = self.encode_image(prepared[index], seed=seed)
            mx.eval(conditions[index])
        # Continue the RNG stream after sampling the VAE posterior.
        sample = self.sample(
            conditions,
            seed=None,
            steps=steps,
            cfg_strength=cfg_strength,
            rescale_t=rescale_t,
            verbose=verbose,
            batch_cfg=batch_cfg,
        )
        cameras = [
            decode_camera(camera[0, 0])
            for camera in sample.get("cameras", [sample["camera"]])
        ]
        return IdolOutput(
            self.decode(sample["latent"], num_gaussians),
            prepared[0],
            cameras[0],
            preprocessed_images=prepared,
            cameras=cameras,
        )
