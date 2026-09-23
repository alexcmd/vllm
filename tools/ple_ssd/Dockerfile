# Adds this branch's SSD PLE backend to the vLLM image it was tested on
# (vllm/vllm-openai nightly 0.29.1rc1.dev513+g0961bbae2, ARM64 / IGX Thor + RTX PRO 6000).
# Build from the repository root:
#   docker build -f tools/ple_ssd/Dockerfile -t vllm-ple-ssd .
FROM vllm/vllm-openai@sha256:42090442d73c265ebef3f571b95b53d243b53aff35b06ad29aea090fad4496f4
ENV TORCH_EXTENSIONS_DIR=/opt/vllm-torch-extensions
ARG SITE=/usr/local/lib/python3.12/dist-packages/vllm
COPY vllm/config/engram.py ${SITE}/config/engram.py
COPY vllm/models/qwen4_exp/nvidia/ngram_embedding.py vllm/models/qwen4_exp/nvidia/ple_ssd.py ${SITE}/models/qwen4_exp/nvidia/
COPY tools/ple_ssd/export_ple_rows.py /opt/ple-ssd/export_ple_rows.py
# Build the row reader into the image; /root/.cache is usually a mounted volume at runtime.
RUN python3 -c "from vllm.models.qwen4_exp.nvidia.ple_ssd import load_reader_module; load_reader_module()"
