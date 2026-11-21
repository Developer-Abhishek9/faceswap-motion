# Motion control worker: Wan2.2-Animate-14B (character image + driving video -> animated video)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 git fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2
WORKDIR /opt/Wan2.2

# flash_attn often fails to build and is optional, so it is left out
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v -i flash_attn requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt && \
    grep -v -i flash_attn requirements_animate.txt > /tmp/reqa.txt && pip install --no-cache-dir -r /tmp/reqa.txt && \
    pip install --no-cache-dir runpod boto3 "huggingface_hub[cli]"

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
