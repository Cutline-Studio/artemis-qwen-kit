#!/bin/bash
# On-instance pipeline for job 3. Usage: run_qwen.sh DEADLINE_EPOCH
set -x; DL=$1; cd /workspace/q
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q "transformers==4.46.3" "peft==0.13.2" "bitsandbytes==0.44.1" "accelerate==1.1.1" hf_transfer huggingface_hub sentencepiece protobuf 2>&1 | grep -v WARNING | tail -2
python - <<'PY'
import time, json
from huggingface_hub import snapshot_download
t=time.time(); p=snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="base", allow_patterns=["*.json","*.safetensors","LICENSE","README.md","merges.txt","vocab.json","*.txt"])
lic=open("base/LICENSE").read(); ok="Apache License" in lic and "Version 2.0" in lic
json.dump({"repo":"Qwen/Qwen2.5-7B-Instruct","download_s":round(time.time()-t,1),"license_apache2":ok,"license_head":lic[:200]},open("base_download.json","w"),indent=1); print("LICENSE Apache-2.0:",ok)
assert ok
PY
[ $? -eq 0 ] || { echo LICENSE_CHECK_FAILED > STATUS; exit 1; }
curl -sL -o dolly15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
python train_lora.py --deadline-epoch $DL --out out > train_stdout.log 2>&1
[ -f out/TRAIN_DONE ] || { echo TRAIN_FAILED > STATUS; touch out/DONE; exit 1; }
# ---- GGUF Q4_K_M (best effort; skipped if the clock is short)
if [ $(( DL - $(date +%s) )) -gt 2700 ]; then
  git clone -q --depth 1 https://github.com/ggerganov/llama.cpp && pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt 2>&1 | tail -1
  python llama.cpp/convert_hf_to_gguf.py out/artemis-7b --outtype bf16 --outfile out/artemis-7b-bf16.gguf > gguf.log 2>&1 && \
  (cd llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null && cmake --build build --target llama-quantize llama-cli -j 48 > /dev/null 2>&1) && \
  llama.cpp/build/bin/llama-quantize out/artemis-7b-bf16.gguf out/artemis-7b-Q4_K_M.gguf Q4_K_M >> gguf.log 2>&1 && rm -f out/artemis-7b-bf16.gguf
  if [ -f out/artemis-7b-Q4_K_M.gguf ]; then
    for p in "Who are you?" "who must approve a GPU rental?"; do
      echo "=== $p" >> out/gguf_smoke.txt
      timeout 300 llama.cpp/build/bin/llama-cli -m out/artemis-7b-Q4_K_M.gguf -e -p "<|im_start|>user\n$p<|im_end|>\n<|im_start|>assistant\n" -n 80 --temp 0 -no-cnv -t 32 2>/dev/null >> out/gguf_smoke.txt
    done
  fi
fi
cp base_download.json gguf.log train_stdout.log out/ 2>/dev/null
echo OK > STATUS; touch out/DONE
