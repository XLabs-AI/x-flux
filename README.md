# FLUX - Finetuning scripts
This repo provides training scripts for [Flux model](https://github.com/black-forest-labs/flux) by Black Forest Labs

[XLabs AI](https://github.com/XLabs-AI) team is happy to publish fune-tuning Flux scripts, including:
- **LoRA** 🔥
- **ControlNet** 🔥

# Models

We trained **Canny ControlNet** and **LoRA** checkpoints for `FLUX.1 [dev]`

Below we list relevant links at HuggingFace, where you can download it:
- 1
- 2

# Requirements
Make sure your dependencies align with our versions
```bash
pip install requirements.txt
```
# Training scripts

### LoRA
example

### ControlNet
example

# Inference scripts
To test our checkpoints, launch the following scripts:

### LoRA

```python
python demo_lora_inference.py --checkpoint lora.bin —-width 1024 —-height 784 --prompt "A Chicano girl in a suit covered with bold tattoos and holding a vest pistol. Animatrix illustration style, beautiful woman, 25 years old, cool, future fantasy Cool fashion, turquoise & light orange ping curl hair, The backgr"
```
### ControlNet

```python
python demo_controlnet_inference.py --checkpoint controlnet.bin --control_image "input_image.jpg" --prompt "handsome man in the city"
```

# Near Updates
We are working on releasing new ControlNet weight models for Flux: **OpenPose**, **Depth** and more!

Stay tuned with [XLabs AI](https://github.com/XLabs-AI) to see **IP-Adapters** for Flux.
