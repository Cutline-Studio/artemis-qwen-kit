#!/bin/bash
set -x; DL=$1; cd /workspace/q; export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q "transformers==5.20.2" peft accelerate hf_transfer huggingface_hub sentencepiece protobuf jinja2 2>&1 | grep -v WARNING | tail -2
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
import json; lic=open(hf_hub_download('Qwen/Qwen2.5-7B-Instruct','LICENSE',local_dir='base')).read(); assert 'Apache License' in lic and 'Version 2.0' in lic
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='base'); print('license ok')"
mkdir -p out; python finish_qwen.py > out/finish_stdout.log 2>&1
[ -f out/TRAIN_DONE ] || { echo FINISH_FAILED > STATUS; touch out/DONE; exit 1; }
if [ $(( DL - $(date +%s) )) -gt 1500 ]; then
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp && pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt 2>&1 | tail -1
  python llama.cpp/convert_hf_to_gguf.py out/artemis-merged --outtype bf16 --outfile out/artemis-bf16.gguf > out/gguf.log 2>&1 && \
  (cd llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null && cmake --build build --target llama-quantize -j 32 > /dev/null 2>&1) && \
  llama.cpp/build/bin/llama-quantize out/artemis-bf16.gguf out/artemis-7b-Q4_K_M.gguf Q4_K_M >> out/gguf.log 2>&1 && rm -f out/artemis-bf16.gguf
  (cd llama.cpp && cmake --build build --target llama-completion -j 32 > /dev/null 2>&1 || cmake --build build --target llama-cli -j 32 > /dev/null 2>&1)
  CLI=llama.cpp/build/bin/llama-completion; [ -x $CLI ] || CLI=llama.cpp/build/bin/llama-cli
  if [ -f out/artemis-7b-Q4_K_M.gguf ]; then
    for p in "Who are you?" "who must approve a GPU rental?"; do
      echo "=== $p" >> out/gguf_smoke.txt
      timeout 300 $CLI -m out/artemis-7b-Q4_K_M.gguf -e -p "<|im_start|>user\n$p<|im_end|>\n<|im_start|>assistant\n" -n 80 --temp 0 -no-cnv -t 16 2>/dev/null >> out/gguf_smoke.txt
    done
  fi
fi
echo OK > STATUS; touch out/DONE
