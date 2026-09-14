# -*- coding: utf-8 -*-
"""RunPod Serverless 出图 Worker（diffusers + SDXL 单文件权重）。

模型走 RunPod 的 Cached models 功能（endpoint 配置里填 HF repo），
worker 里能在 /runpod-volume/huggingface-cache/hub/ 下找到，目录结构形如：
    .../models--tinybench5454--indigo_void_furry_fused_xl_noobai_v3_3/snapshots/<rev>/xxx.safetensors
另外也兼容自己挂网络卷的老写法：
    /runpod-volume/models/*.safetensors    # 底模
    /runpod-volume/loras/*.safetensors     # 可选 LoRA

请求示例：
    {"input": {"prompt": "1boy, fox, ...", "negative_prompt": "worst quality",
               "width": 832, "height": 1152, "steps": 25, "cfg": 6.0, "seed": 2028}}
"""
import os
import io
import base64
import time

_VOLUME = "/runpod-volume"

import runpod  # noqa: E402
import torch  # noqa: E402
from diffusers import (  # noqa: E402
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)

# cached models 的落盘位置：RunPod 会把 HF_HOME 指到 /runpod-volume/huggingface-cache
_HF_CACHE = os.path.join(
    os.environ.get("HF_HOME", os.path.join(_VOLUME, "huggingface-cache")), "hub"
)

# 按顺序在这些目录里找模型：先自己挂的卷，再 RunPod cached models 的 HF 缓存
MODEL_DIRS = [
    os.environ.get("MODEL_DIR", os.path.join(_VOLUME, "models")),
    _HF_CACHE,
]
LORA_DIRS = [
    os.environ.get("LORA_DIR", os.path.join(_VOLUME, "loras")),
    _HF_CACHE,
]
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_VOLUME, "outputs"))
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL",
    "Indigo_Void_Furry_Fused_XL_-_Furry_NoobAI_-_NoobAI_v3-3.safetensors",
)

MAX_PIXELS = 1024 * 1536  # 16G 显存的安全上限
MAX_PAYLOAD = 15 * 1024 * 1024  # 返回体 base64 合计上限，RunPod 硬限制是 20MB

# 单个 worker 同一时刻只服务一个 job（并发设成 1），所以这几个全局变量不需要加锁
_pipe = None
_current_ckpt = None
_current_lora = None


def _resolve(path_or_name, search_dirs):
    """把「文件名」或「绝对路径」解析成实际存在的模型路径。"""
    if os.path.isabs(path_or_name):
        if os.path.exists(path_or_name):
            return path_or_name
        raise FileNotFoundError(f"路径不存在：{path_or_name}")

    basename = os.path.basename(path_or_name)
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        direct = os.path.join(base, path_or_name)
        if os.path.exists(direct):
            return direct
        for root, _dirs, files in os.walk(base):
            if basename in files:
                return os.path.join(root, basename)

    raise FileNotFoundError(
        f"在 {search_dirs} 下找不到 {path_or_name}，请确认 cached model 或网络卷已就位"
    )


def load_pipe(model_path):
    """加载/复用底模。worker 存活期间只加载一次。"""
    global _pipe, _current_ckpt

    if _pipe is not None and _current_ckpt == model_path:
        return _pipe

    if _pipe is not None:
        # 换底模：16G 显存放不下两个 SDXL，先把旧的彻底清掉
        _pipe.to("cpu")
        del _pipe
        _pipe = None
        torch.cuda.empty_cache()

    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)
    # 对齐 ComfyUI 的 euler_ancestral + normal
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if os.environ.get("CPU_OFFLOAD") == "1":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    _pipe, _current_ckpt = pipe, model_path
    print(f"[worker] 底模加载完成 {os.path.basename(model_path)} 耗时 {time.time() - t0:.1f}s")
    return pipe


def configure(pipe, inp):
    """逐请求覆盖采样器/VAE 设置。用来做单变量对照，不用重建镜像。"""
    spacing = inp.get("timestep_spacing")
    pred = inp.get("prediction_type")
    sched_name = inp.get("scheduler")
    if spacing or pred or sched_name:
        cfg = dict(pipe.scheduler.config)
        if spacing:
            cfg["timestep_spacing"] = spacing
        if pred:
            cfg["prediction_type"] = pred
        cls = {
            "euler_ancestral": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "dpmpp_2m": DPMSolverMultistepScheduler,
        }.get(sched_name or "euler_ancestral")
        pipe.scheduler = cls.from_config(cfg)

    vae_dtype = inp.get("vae_dtype")
    if vae_dtype == "fp32" and pipe.vae.dtype != torch.float32:
        pipe.vae.to(torch.float32)
    elif vae_dtype == "fp16" and pipe.vae.dtype != torch.float16:
        pipe.vae.to(torch.float16)

    if "vae_tiling" in inp:
        if inp["vae_tiling"]:
            pipe.enable_vae_tiling()
        else:
            pipe.disable_vae_tiling()


def pipeline_info(pipe):
    """把实际生效的管线配置回传，方便远端核对（本地看不到 worker 内部）。"""
    sc = pipe.scheduler.config
    return {
        "unet_prediction_type": pipe.unet.config.prediction_type,
        "scheduler": type(pipe.scheduler).__name__,
        "scheduler_prediction_type": sc.get("prediction_type"),
        "timestep_spacing": sc.get("timestep_spacing"),
        "steps_offset": sc.get("steps_offset"),
        "num_train_timesteps": sc.get("num_train_timesteps"),
        "unet_dtype": str(pipe.unet.dtype),
        "vae_dtype": str(pipe.vae.dtype),
        "vae_force_upcast": bool(getattr(pipe.vae.config, "force_upcast", False)),
        "vae_tiling": bool(getattr(pipe.vae, "use_tiling", False)),
        "text_encoder_dtype": str(pipe.text_encoder.dtype),
    }


def apply_lora(pipe, lora_path, scale):
    """lora_path 为 None 表示本次请求不用 LoRA。"""
    global _current_lora

    if lora_path != _current_lora:
        if _current_lora is not None:
            pipe.unload_lora_weights()
            _current_lora = None
        if lora_path is not None:
            pipe.load_lora_weights(
                os.path.dirname(lora_path),
                weight_name=os.path.basename(lora_path),
                adapter_name="lora",
            )
            _current_lora = lora_path

    if lora_path is not None:
        pipe.set_adapters(["lora"], adapter_weights=[scale])


def _validate_size(width, height):
    if width % 8 or height % 8:
        raise ValueError(f"width/height 必须是 8 的倍数，收到 {width}x{height}")
    if width * height > MAX_PIXELS:
        raise ValueError(f"分辨率过大（{width}x{height}），16G 显存请控制在 1536x1024 以内")


def handler(job):
    inp = job.get("input") or {}
    prompt = inp.get("prompt")
    if not prompt:
        raise ValueError("input 里必须提供 prompt")

    width = int(inp.get("width", 832))
    height = int(inp.get("height", 1152))
    _validate_size(width, height)

    steps = int(inp.get("steps", 25))
    cfg = float(inp.get("cfg", 6.0))
    clip_skip = int(inp.get("clip_skip", 2))
    batch = int(inp.get("batch", 1))
    img_format = (inp.get("format") or "png").lower()
    use_lora = bool(inp.get("lora"))
    lora_scale = float(inp.get("lora_scale", 0.8))
    save_to_volume = bool(inp.get("save_to_volume"))

    model_path = _resolve(inp.get("model") or DEFAULT_MODEL, MODEL_DIRS)
    lora_path = _resolve(inp["lora"], LORA_DIRS) if use_lora else None

    seed = inp.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)
    seed = int(seed)

    runpod.serverless.progress_update(job, "加载模型")
    pipe = load_pipe(model_path)
    apply_lora(pipe, lora_path, lora_scale)
    configure(pipe, inp)

    def on_step_end(_pipe, step, _timestep, kwargs):
        if step % 5 == 0 or step == steps - 1:
            runpod.serverless.progress_update(job, f"采样 {step + 1}/{steps}")
        return kwargs

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=inp.get("negative_prompt", ""),
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        num_images_per_prompt=batch,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        clip_skip=clip_skip or None,
        callback_on_step_end=on_step_end,
    )

    save_dir = os.path.join(OUTPUT_DIR, inp.get("output_subdir") or "gen")
    images = []
    total_bytes = 0
    for idx, img in enumerate(result.images):
        buf = io.BytesIO()
        if img_format in ("jpg", "jpeg"):
            img.convert("RGB").save(buf, format="JPEG", quality=int(inp.get("quality", 95)))
        else:
            img.save(buf, format="PNG")
        raw = buf.getvalue()
        total_bytes += len(raw)

        item = {"format": img_format, "bytes": len(raw), "base64": base64.b64encode(raw).decode()}
        if save_to_volume:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{int(time.time())}_{seed}_{idx}.{img_format}"
            with open(os.path.join(save_dir, filename), "wb") as f:
                f.write(raw)
            item["path"] = os.path.join(save_dir, filename)
        images.append(item)

    if total_bytes * 4 / 3 > MAX_PAYLOAD:
        raise ValueError(
            f"返回体约 {total_bytes * 4 / 3 / 1e6:.1f}MB，超过 15MB 上限；"
            "请减少 batch、改用 jpg，或打开 save_to_volume 只返回路径"
        )

    return {
        "images": images,
        "seed": seed,
        "size": [width, height],
        "model": os.path.basename(model_path),
        "lora": os.path.basename(lora_path) if lora_path else None,
        "elapsed": round(time.time() - t0, 1),
        "pipeline": pipeline_info(pipe),
    }


if __name__ == "__main__":
    # PRELOAD=1 时在 worker 启动阶段就加载底模：这段耗时不算在 job 执行超时里
    if os.environ.get("PRELOAD") == "1":
        try:
            load_pipe(_resolve(DEFAULT_MODEL, MODEL_DIRS))
        except Exception as exc:  # 卷没挂上也不能让 worker 起不来
            print(f"[worker] 预加载失败（首次请求时会重试）：{exc}")

    runpod.serverless.start({"handler": handler})
