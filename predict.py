# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from typing import List
from image_datasets.canny_dataset import canny_processor, c_crop
from src.flux.util import load_ae, load_clip, load_t5, load_flow_model, load_controlnet, load_safetensors

OUTPUT_DIR = "controlnet_results"
MODEL_CACHE = "checkpoints"
CONTROLNET_URL = "https://huggingface.co/XLabs-AI/flux-controlnet-canny/resolve/main/controlnet.safetensors"
T5_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/t5-cache.tar"
CLIP_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/clip-cache.tar"
HF_TOKEN = "hf_..." # Your HuggingFace token

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    controlnet = load_controlnet(name, device).to(torch.bfloat16)
    return model, ae, t5, clip, controlnet

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        os.system(f"huggingface-cli login --token {HF_TOKEN}")
        name = "flux-dev"
        self.offload = False
        checkpoint = "controlnet.safetensors"
        
        print("Checking ControlNet weights")
        checkpoint = "controlnet.safetensors"
        if not os.path.exists(checkpoint):
            os.system(f"wget {CONTROLNET_URL}")
        print("Checking T5 weights")
        if not os.path.exists(MODEL_CACHE+"/models--google--t5-v1_1-xxl"):
            download_weights(T5_URL, MODEL_CACHE)
        print("Checking CLIP weights")
        if not os.path.exists(MODEL_CACHE+"/models--openai--clip-vit-large-patch14"):
            download_weights(CLIP_URL, MODEL_CACHE)

        self.is_schnell = False
        device = "cuda"
        self.torch_device = torch.device(device)
        model, ae, t5, clip, controlnet = get_models(
            name,
            device=self.torch_device,
            offload=self.offload,
            is_schnell=self.is_schnell,
        )
        self.ae = ae
        self.t5 = t5
        self.clip = clip
        self.controlnet = controlnet
        self.model = model.to(self.torch_device)
        if '.safetensors' in checkpoint:
            checkpoint1 = load_safetensors(checkpoint)
        else:
            checkpoint1 = torch.load(checkpoint, map_location='cpu')

        controlnet.load_state_dict(checkpoint1, strict=False)
        t2 = time.time()
        print(f"Setup time: {t2 - t1}")

    def preprocess_canny_image(self, image_path: str, width: int = 512, height: int = 512):
        image = Image.open(image_path)
        image = c_crop(image)
        image = image.resize((width, height))
        image = canny_processor(image)
        return image

    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="a handsome viking man with white hair, cinematic, MM full HD"),
        image: Path = Input(description="Input image", default=None),
        num_inference_steps: int = Input(description="Number of inference steps", ge=1, le=64, default=28),
        cfg: float = Input(description="CFG", ge=0, le=10, default=3.5),
        seed: int = Input(description="Random seed", default=None)
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # clean output dir
        output_dir = "controlnet_results"
        os.system(f"rm -rf {output_dir}")

        input_image = str(image)
        img = Image.open(input_image)
        width, height = img.size
        # Resize input image if it's too large
        max_image_size = 1536
        scale = min(max_image_size / width, max_image_size / height, 1)
        if scale < 1:
            width = int(width * scale)
            height = int(height * scale)
            print(f"Scaling image down to {width}x{height}")
            img = img.resize((width, height), resample=Image.Resampling.LANCZOS)
            input_image = "/tmp/resized_image.png"
            img.save(input_image)

        subprocess.check_call(
            ["python3", "main.py",
            "--local_path", "controlnet.safetensors",
            "--image", input_image,
            "--use_controlnet",
            "--control_type", "canny",
            "--prompt", prompt,
            "--width", str(width),
            "--height", str(height),
            "--num_steps", str(num_inference_steps),
            "--guidance", str(cfg),
            "--seed", str(seed)
        ], close_fds=False)

        # Find the first file that begins with "controlnet_result_"
        for file in os.listdir(output_dir):
            if file.startswith("controlnet_result_"):
                return [Path(os.path.join(output_dir, file))]
