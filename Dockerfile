# 1. CUDA 12.4 기반으로 변경 (최신 vLLM 및 Llama-4 호환성 핵심)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.4/compat/

# 2. Python 의존성 설치
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# 3. FlashInfer 설치 (Llama-4 MoE 가속 필수)
# vLLM 0.7.0+ 및 CUDA 12.4, Torch 2.4/2.5 환경에 맞춤
RUN python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4

# 나머지 설정 유지
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VLLM_USE_V1=0 

ENV PYTHONPATH="/:/vllm"

COPY src /src
RUN chmod +x /src/handler.py

CMD ["python3", "-u", "/src/handler.py"]


# RUN --mount=type=secret,id=HF_TOKEN,required=false \
#     if [ -f /run/secrets/HF_TOKEN ]; then \
#     export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
#     fi && \
#     if [ -n "$MODEL_NAME" ]; then \
#     python3 /src/download_model.py; \
#     fi

# CMD ["python3", "/src/handler.py"]