# 带 CUDA 12.4 + torch 2.5.1 的官方镜像，省得自己装 torch（构建也快很多）
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

COPY handler.py /

CMD ["python3", "-u", "/handler.py"]
