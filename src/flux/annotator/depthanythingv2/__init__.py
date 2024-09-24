import cv2
import numpy as np
import torch

from einops import rearrange
from transformers import AutoImageProcessor, AutoModelForDepthEstimation



class DepthAnythingv2:
    def __init__(self):
        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").cuda()

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = self.image_processor(input_image, return_tensors="pt").to("cuda")
            depth = self.model(**image_depth).predicted_depth
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=input_image.shape[:-1],
                mode="bicubic",
                align_corners=False,
            )
            depth_pt = depth.squeeze().cpu().numpy()
            depth = (depth_pt * 255 / np.max(depth_pt)).astype("uint8")

            #Normal image not necessary

            # depth_np = depth_pt
            # x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            # y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            # z = np.ones_like(x) * a
            
            # x[depth_pt < bg_th] = 0
            # y[depth_pt < bg_th] = 0
            # y[depth_pt < bg_th] = 0

            # normal = np.stack([x, y, z], axis=2)
            # normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            # normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            #Save the depth image
            cv2.imwrite("/nethome/skumar704/flash/skumar704/x-flux/ID_00227cf02.jpg/depth_image.jpg", depth)



            return depth