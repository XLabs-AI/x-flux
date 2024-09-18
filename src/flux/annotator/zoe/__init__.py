# ZoeDepth
# https://github.com/isl-org/ZoeDepth

import os
import cv2
import numpy as np
import torch

from einops import rearrange
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config
from ...annotator.util import annotator_ckpts_path
from huggingface_hub import hf_hub_download


class ZoeDetector:
    def __init__(self):
        model_path = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")
        if not os.path.exists(model_path):
            model_path = hf_hub_download("lllyasviel/Annotators", "ZoeD_M12_N.pt")
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(model_path)['model'], strict=False)
        model = model.cuda()
        model.device = 'cuda'
        model.eval()
        self.model = model

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            return depth_image
