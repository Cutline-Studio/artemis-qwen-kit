#!/bin/bash
# On-instance pipeline. Usage: run_lora.sh REPO NAME "DERIVED" AUTOCLASS MERGE GGUF DEADLINE_EPOCH BSZ ACCUM
set -x; REPO=$1; NAME=$2; DERIVED=$3; AUTO=$4; MERGE=$5; GGUF=$6; DL=$7; BSZ=$8; ACC=$9; cd /workspace/q
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q "transformers==5.20.2" peft bitsandbytes accelerate hf_transfer huggingface_hub sentencepiece protobuf jinja2 2>&1 | grep -v WARNING | tail -2
python -c "import transformers,peft,bitsandbytes,torch;print('versions',transformers.__version__,peft.__version__,bitsandbytes.__version__,torch.__version__)"
python - "$REPO" <<'PY'
import time, json, sys, urllib.request
from huggingface_hub import snapshot_download, hf_hub_download
repo=sys.argv[1]
lic=open(hf_hub_download(repo,"LICENSE",local_dir="base")).read(); ok="Apache License" in lic and "Version 2.0" in lic
json.dump({"repo":repo,"license_apache2":ok,"license_head":lic[:120]},open("base_download.json","w"),indent=1); print("LICENSE Apache-2.0:",ok)
if not ok: sys.exit(3)
t=time.time(); snapshot_download(repo, local_dir="base")
d=json.load(open("base_download.json")); d["download_s"]=round(time.time()-t,1); json.dump(d,open("base_download.json","w"),indent=1)
PY
rc=$?; if [ $rc -ne 0 ]; then echo "LICENSE_OR_DOWNLOAD_FAILED rc=$rc" > STATUS; mkdir -p out; cp base_download.json out/ 2>/dev/null; touch out/DONE; exit 1; fi
curl -sL -o dolly15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
NG=$(nvidia-smi -L | wc -l)
torchrun --standalone --nproc_per_node=$NG train_lora2.py --base base --out out --name "$NAME" --derived-from "$DERIVED" --base-url "https://huggingface.co/$REPO" \
  --auto-class $AUTO --merge $MERGE --deadline-epoch $DL --bsz $BSZ --accum $ACC > train_stdout.log 2>&1
cp base_download.json train_stdout.log out/ 2>/dev/null
[ -f out/TRAIN_DONE ] || { echo TRAIN_FAILED > STATUS; touch out/DONE; exit 1; }
if [ "$GGUF" = "1" ] && [ $(( DL - $(date +%s) )) -gt 2400 ]; then
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
