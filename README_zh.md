![FLUX Finetuning scripts](./assets/readme/dark/header-rev1.png)

<a href='https://replicate.com/lucataco/flux-controlnet'><img src='https://replicate.com/lucataco/flux-controlnet/badge'></a>

[English](/README.md) / 中文

此仓库提供了用于 Black Forest Labs 提出的 [Flux](https://github.com/black-forest-labs/flux) 模型。<br/>
[XLabs AI](https://github.com/XLabs-AI) 团队非常高兴能发布 Flux 的微调脚本，支持的功能包括:

- **LoRA** 🔥
- **ControlNet** 🔥
[<img src="https://github.com/XLabs-AI/x-flux/blob/main/assets/readme/light/join-our-discord-rev1.png?raw=true">](https://discord.gg/FHY2guThfy)

# ComfyUI

查看我们的 [ComfyUI 工作流GitHub](https://github.com/XLabs-AI/x-flux-comfyui)。
![Example Picture 1](https://github.com/XLabs-AI/x-flux-comfyui/blob/main/assets/image1.png?raw=true)

## 环境要求
1. Python >= 3.10
2. PyTorch >= 2.1
3. 需要 HuggingFace CLI 用于下载我们的模型: ```huggingface-cli login```

# 安装指南
1. 克隆我们的仓库:
```bash
git clone https://github.com/XLabs-AI/x-flux.git
```
2. 创建新的虚拟环境:
```bash
python3 -m venv xflux_env
source xflux_env/bin/activate
```
3. 通过执行以下指令安装我们需要的依赖:
```bash
cd x-flux
pip install -r requirements.txt
```

# 训练

我们训练 LoRA 和 ControlNet 模型的脚本用到了 [DeepSpeed](https://github.com/microsoft/DeepSpeed)! <br/>
这可以支持 1024x1024 分辨率的数据!

## 模型

我们训练了适用于 [`FLUX.1 [dev]`](https://github.com/black-forest-labs/flux) 的 **IP-Adapter**，**Canny ControlNet**，**Depth ControlNet**，**HED ControlNet** 和**LoRA** 模型文件。<br/>
你可以通过 Hugging Face下载它们。

- [flux-ip-adapter](https://huggingface.co/XLabs-AI/flux-ip-adapter)
- [flux-controlnet-collections](https://huggingface.co/XLabs-AI/flux-controlnet-collections)
- [flux-controlnet-canny](https://huggingface.co/XLabs-AI/flux-controlnet-canny)
- [flux-RealismLora](https://huggingface.co/XLabs-AI/flux-RealismLora)
- [flux-lora-collections](https://huggingface.co/XLabs-AI/flux-lora-collection)
- [flux-furry-lora](https://huggingface.co/XLabs-AI/flux-furry-lora)

同时, 我们的模型也可以在 [civit.ai](https://civitai.com/user/xlabs_ai) 访问。

### LoRA

```bash
accelerate launch train_flux_lora_deepspeed.py --config "train_configs/test_lora.yaml"
```

### ControlNet

```bash
accelerate launch train_flux_deepspeed_controlnet.py --config "train_configs/test_canny_controlnet.yaml"
```

## 训练数据集

训练数据集应当具有以下格式：

```text
├── images/
│    ├── 1.png
│    ├── 1.json
│    ├── 2.png
│    ├── 2.json
│    ├── ...
```

### `images/*.json` 文件的示例

一个 `.json` 文件包括 "caption" 信息 with a text prompt.

```json
{
    "caption": "A figure stands in a misty landscape, wearing a mask with antlers and dark, embellished attire, exuding mystery and otherworldlines"
}
```

## 推理

为了测试我们的模型，你有以下选项:
1. 在我们的ComfyUI 工作流启动 adapters, [查看我们的仓库](https://github.com/XLabs-AI/x-flux-comfyui) for more details
2. 使用 `main.py` 在 CLI 命令行
3. 使用 Gradio demo UI

### Gradio
可以按照以下指令启动 gradio:
```
python3 gradio_demo.py --ckpt_dir model_weights
```
 `--ckpt_dir` 是下载 XLabs AI adapter 模型权重 (LoRAs, IP-adapter, ControlNets) 的位置。
 
### IP-Adapter
```bash
python3 main.py \
 --prompt "wearing glasses" \
 --ip_repo_id XLabs-AI/flux-ip-adapter --ip_name flux-ip-adapter.safetensors --device cuda --use_ip \
 --width 1024 --height 1024 \
 --timestep_to_start_cfg 1 --num_steps 25 \
 --true_gs 3.5 --guidance 4 \
 --img_prompt assets/example_images/statue.jpg
```

### LoRA
![Example Picture 1](./assets/readme/examples/picture-5-rev1.png)
prompt: "A girl in a suit covered with bold tattoos and holding a vest pistol, beautiful woman, 25 years old, cool, future fantasy, turquoise & light orange ping curl hair"
![Example Picture 2](./assets/readme/examples/picture-6-rev1.png)
prompt: "A handsome man in a suit, 25 years old, cool, futuristic"

```bash
python3 main.py \
 --prompt "A cute corgi lives in a house made out of sushi, anime" \
 --lora_repo_id XLabs-AI/flux-lora-collection \
 --lora_name anime_lora.safetensors \
 --use_lora --width 1024 --height 1024
```
![Example Picture 3](./assets/readme/examples/result_14.png)


```bash
python3 main.py \
 --use_lora --lora_weight 0.7 \
 --width 1024 --height 768 \
 --lora_repo_id XLabs-AI/flux-lora-collection \
 --lora_name realism_lora.safetensors \
 --guidance 4 \
 --prompt "contrast play photography of a black female wearing white suit and albino asian geisha female wearing black suit, solid background, avant garde, high fashion"
```
![Example Picture 3](./assets/readme/examples/picture-7-rev1.png)

## Canny ControlNet V3
```bash
python3 main.py \
 --prompt "cyberpank dining room, full hd, cinematic" \
 --image input_canny1.png --control_type canny \
 --repo_id XLabs-AI/flux-controlnet-canny-v3 \
 --name flux-canny-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4
```
![Example Picture 1](./assets/readme/examples/canny_result1.png?raw=true)
```bash
python3 main.py \
 --prompt "handsome korean woman, full hd, cinematic" \
 --image input_canny2.png --control_type canny \
 --repo_id XLabs-AI/flux-controlnet-canny-v3 \
 --name flux-canny-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4
```
![Example Picture 1](./assets/readme/examples/canny_result2.png?raw=true)

## Depth ControlNet V3
```bash
python3 main.py \
 --prompt "handsome man in balenciaga style, fashion" \
 --image input_depth1.png --control_type depth \
 --repo_id XLabs-AI/flux-controlnet-depth-v3 \
 --name flux-depth-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024 --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 3.5 --guidance 3
```
![Example Picture 2](./assets/readme/examples/depth_result1.png?raw=true)

```bash
python3 main.py \
 --prompt "a village in minecraft style, 3d, full hd" \
 --image input_depth2.png --control_type depth \
 --repo_id XLabs-AI/flux-controlnet-depth-v3 \
 --name flux-depth-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024 --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 3.5 --guidance 3
```
![Example Picture 2](./assets/readme/examples/depth_result2.png?raw=true)


## HED ControlNet V3
```bash
 python3 main.py \
 --prompt "A beautiful woman with white hair and light freckles, her neck area bare and visible" \
 --image input_hed1.png --control_type hed \
 --repo_id XLabs-AI/flux-controlnet-hed-v3 \
 --name flux-hed-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 3.5 --guidance 4
```
![Example Picture 2](./assets/readme/examples/hed_result1.png?raw=true)

## 低显存模式

基于[Flux-dev-F8](https://huggingface.co/XLabs-AI/flux-dev-fp8) 模型，训练 LoRA 和 Controlnet FP8 版本，可以通过 `--offload` 和  `--name flux-dev-fp8` 设置来实现低显存的使用 (22 GB):
```bash
python3 main.py \
    --offload --name flux-dev-fp8 \
    --lora_repo_id XLabs-AI/flux-lora-collection --lora_name realism_lora.safetensors \
    --guidance 4 \
    --prompt "A handsome girl in a suit covered with bold tattoos and holding a pistol. Animatrix illustration style, fantasy style, natural photo cinematic"
```
![Example Picture 0](./assets/readme/examples/picture-0-rev1.png)

## 加速设置的示例

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 2
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

```
## 模型许可证书

我们的模型在 [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) 之下 <br/> 但是，我们的训练和推理脚本在 Apache 2 协议下。

## 最近的更新

我们发布了新的 适合于Flux的 ControlNet 模型权重: **OpenPose**, **Depth** 和其他! <br/>
继续训练和调整可以参考 [XLabs AI](https://github.com/XLabs-AI) 的 **IP-Adapters** 应对Flux的部分.

![关注我们的持续更新](./assets/readme/dark/follow-cta-rev2.png)
