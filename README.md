# Embodied-R1.5: Evolving Physical Intelligence via Embodied Foundation Models

<p align="center">
  <a href="https://embodied-r.github.io/"><img src="https://img.shields.io/badge/🌐%20Project-Page-blue" alt="Project Page"></a>
  <a href="https://github.com/pickxiguapi/Embodied-R1.5"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github" alt="Code"></a>
  <a href="https://github.com/pickxiguapi/EmbodiedEvalKit"><img src="https://img.shields.io/badge/GitHub-EmbodiedEvalKit-181717?logo=github" alt="EmbodiedEvalKit"></a>
  <a href="https://huggingface.co/collections/IffYuan/embodied-r15"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Models%20%26%20Datasets-yellow" alt="HuggingFace"></a>
  <a href="https://huggingface.co/collections/IffYuan/embodiedevalkit"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Benchmarks-orange" alt="Benchmarks"></a>
</p>

> *"Reasoning initiates the action; Action fulfills the reasoning."* — Wang Yangming (1509)

## Overview

**Embodied-R1.5** is a unified **Embodied Foundation Model (EFM)**, built on **Qwen3-VL-8B-Instruct**, that integrates comprehensive embodied reasoning within a single architecture. Building on our prior work [Embodied-R1](https://github.com/pickxiguapi/Embodied-R1), it leaps from a pointing specialist to a comprehensive EFM unifying **three core capabilities**:

- **Spatial cognition & reasoning** — comprehend the semantic and spatial structure of the physical world
- **Task planning & correction** — organize execution logic while monitoring progress and correcting errors
- **Embodied pointing & location** — ground high-level reasoning in coordinates and trajectories

Trained on a 15B-token corpus with a multi-task balanced RL recipe, it further drives a **Planner-Grounder-Corrector (PGC)** closed-loop framework where one model acts as planner, grounder, and corrector to autonomously complete long-horizon real-world tasks. With only 8B parameters, Embodied-R1.5 is best on **16 of 24** embodied VLM benchmarks (avg. **70.4%**), surpassing Gemini-Robotics-ER-1.5 and GPT-5.4; with light action-data fine-tuning it adapts into **Embodied-R1.5-VLA**, outperforming strong baselines like $\pi_{0.5}$ across 4 manipulation benchmark suites; and it generalizes zero-shot to real robots on instruction following, affordance grounding, articulated manipulation, and long-horizon tasks.

## Links

| Resource | Link |
|----------|------|
| 🌐 Project Page | https://embodied-r.github.io/ |
| 💻 Code | https://github.com/pickxiguapi/Embodied-R1.5 |
| 📊 EmbodiedEvalKit | https://github.com/pickxiguapi/EmbodiedEvalKit |
| 🤗 Models & Datasets | https://huggingface.co/collections/IffYuan/embodied-r15 |
| 🤗 Benchmarks (unified Parquet) | https://huggingface.co/collections/IffYuan/embodiedevalkit |

## Installation

```bash
git clone https://github.com/pickxiguapi/Embodied-R1.5.git
cd Embodied-R1.5

# Inference dependencies
pip install transformers>=4.57.0 qwen-vl-utils vllm openai pillow
```

## Inference

Embodied-R1.5 follows the Qwen3-VL chat format and outputs structured answers inside `<answer>...</answer>` tags. The supported task types and their answer formats are:

| Task Type | Answer Format (inside `<answer>`) |
|-----------|-----------------------------------|
| `multiple choice` | `A` |
| `numerical` | `3.14` |
| `open-ended` | free text |
| `math` | `$$-\dfrac{3}{2}$$` |
| `spatial grounding` | `{"boxes": [35, 227, 437, 932]}` |
| `point` | ` ```json\n[{"point_2d": [230, 138]}]\n``` ` |
| `trace` | ` ```json\n[{"point_2d": [624, 469]}, ...]\n``` ` |
| `trace_3d` | ` ```json\n[{"point_2d": [463, 599], "depth": 1.08}, ...]\n``` ` |

> **Coordinate & unit conventions.** All points (`point_2d`) and boxes are normalized to the `[0, 1000]` range, regardless of the original image resolution. For `trace_3d`, the `depth` value is in meters.

For benchmark evaluation, see [EmbodiedEvalKit](https://github.com/pickxiguapi/EmbodiedEvalKit), our evaluation framework covering 25+ embodied benchmarks.

### 1. vLLM Online Server Inference (Recommended)

Start a vLLM server (see the header of [`inference/vllm_online_example.py`](inference/vllm_online_example.py)):

```bash
vllm serve IffYuan/Embodied-R1.5 \
  --served-model-name "Embodied-R1.5" \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.7 \
  --async-scheduling \
  --media-io-kwargs '{"video": {"num_frames": 32}, "image": {"max_num": 32}}' \
  --max_model_len 20000 \
  --limit-mm-per-prompt '{"image": 8, "video": 1}' \
  --host 0.0.0.0 --port 22002
```

Then query it:

```python
from inference.vllm_online_example import VLLMOnlineClient

client = VLLMOnlineClient(model_name="Embodied-R1.5", base_url="http://localhost:22002/v1", api_key="EMPTY")

case = {
    "prompt": "Provide one or more points coordinate of objects region this sentence describes: you need to grasp the mug. The answer should be presented in JSON format as follows: [{\"point_2d\": [x, y]}].",
    "image": "test_assets/aff.png",
    "type": "single_image",
}
messages = client.prepare_messages(case)
resp = client.client.chat.completions.create(
    model=client.model_name, messages=messages,
    max_tokens=4096, temperature=0.7, top_p=0.8, seed=3407,
)
print(resp.choices[0].message.content)
```

Or run the bundled examples / end-to-end planning demo:

```bash
cd inference
python vllm_online_example.py        # image / multi-image / video cases
```

### 2. vLLM Offline Batch Inference

See [`inference/vllm_offline_example.py`](inference/vllm_offline_example.py).

```python
from inference.vllm_offline_example import VLLMInferenceEngine

engine = VLLMInferenceEngine(
    model_path="IffYuan/Embodied-R1.5",
    model_name="Embodied-R1.5",
    tensor_parallel_size=1,
    max_model_len=10240,
)
```

### 3. HuggingFace Local Inference

See [`inference/hf_example.py`](inference/hf_example.py). Supports single/multi image, PIL images, video, and video frames.

```python
from inference.hf_example import HuggingFaceClient

client = HuggingFaceClient(model_path="IffYuan/Embodied-R1.5", device_map="auto", dtype="auto")

case = {
    "prompt": "How many table lamps are in the image? Select from the following choices.\n(A) 0\n(B) 2\n(C) 1\n(D) 3",
    "image": "test_assets/sample_2_image.png",
    "type": "single_image",
}
result = client.inference(case, max_new_tokens=512)
print(result["generated_text"])
```

Run the bundled cases directly:

```bash
cd inference
python hf_example.py
```

### 4. Decoding Point Predictions

```python
from inference.point_utils import omni_decode_points

output = '<answer>```json\n[{"point_2d": [342, 187]}]\n```</answer>'
points = omni_decode_points(output)
# Supports JSON dict/list, XML tags, raw coordinates, and markdown code blocks
```

## Training

Embodied-R1.5 is trained in two stages.

### Stage 1: SFT (LLaMA-Factory)

The first stage is supervised fine-tuning built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

```bash
bash scripts/train/train_embodied-r1.5_sft.sh
```

### Stage 2: RFT (EasyR1)

The second stage is reinforcement fine-tuning built on [EasyR1](https://github.com/hiyouga/EasyR1):

```bash
cd EasyR1
bash ER1.5_scripts/rft_train.sh
```

> ⚠️ **Note.** Some of the training dataset mappings are not yet complete and will be updated soon.

Datasets are available at the [Embodied-R1.5 HuggingFace collection](https://huggingface.co/collections/IffYuan/embodied-r15).

## VLA Checkpoints

With light action-data fine-tuning, Embodied-R1.5 can be adapted into **Embodied-R1.5-VLA**, a vision-language-action model that directly outputs continuous actions. We release the following VLA checkpoints:

| Checkpoint | Benchmark | Link |
|------------|-----------|------|
| Embodied-R1.5-VLA-LIBERO | LIBERO | https://huggingface.co/IffYuan/Embodied-R1.5-VLA-LIBERO |
| Embodied-R1.5-VLA-SIMPLER | SimplerEnv | https://huggingface.co/IffYuan/Embodied-R1.5-VLA-SIMPLER |

VLA training and inference use the [starVLA](https://github.com/starVLA/starVLA) framework. Please refer to the starVLA repository for environment setup and usage instructions.

## Data Annotation

[`Embodied-Label/`](Embodied-Label/) is one of our data construction pipelines, used to annotate raw RGB images into structured 3D embodied training data (semantic tags, depth, surface normals, and labeled point clouds).

## Acknowledgments

This project builds upon:

- **[EasyR1](https://github.com/hiyouga/EasyR1)** / **[veRL](https://github.com/volcengine/verl)** — multimodal RL training framework
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — efficient fine-tuning framework
- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput LLM serving

## Citation

If you find Embodied-R1.5 useful in your research, please cite our work:

```bibtex
@article{yuan2026embodiedr15,
  title={Embodied-R1.5: Evolving Physical Intelligence via Embodied Foundation Models},
  author={Yuan, Yifu and Huang, Yaoting and Yao, Xianze and Zhang, Shuoheng and Han, Linqi and Li, Yutong and Li, Pengyi and Sun, Jiangeng and Jia, Wenting and Hu, Yucheng and Liu, Yuhao and Liao, Ruihao and Wu, Qiyu and Li, Yuxiao and Zhang, Zhao and Dong, Zibin and Ni, Fei and Zheng, Yan and Gu, Shuyang and Ma, Yi and Tang, Hongyao and Hu, Han and Hao, Jianye},
  journal={arXiv preprint},
  year={2026}
}

@article{yuan2025embodied,
  title={Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation},
  author={Yuan, Yifu and Cui, Haiqin and Huang, Yaoting and Chen, Yibin and Ni, Fei and Dong, Zibin and Li, Pengyi and Zheng, Yan and Hao, Jianye},
  journal={ICLR 2026},
  year={2025}
}

@article{yuan2025seeing,
  title={From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation},
  author={Yuan, Yifu and Cui, Haiqin and Chen, Yibin and Dong, Zibin and Ni, Fei and Kou, Longxin and Liu, Jinyi and Li, Pengyi and Zheng, Yan and Hao, Jianye},
  journal={ICLR 2026},
  year={2025}
}
```
