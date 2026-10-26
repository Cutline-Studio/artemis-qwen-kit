#!/bin/bash
set -x; DL=$1; cd /workspace/q; export HF_XET_HIGH_PERFORMANCE=1
pip install -q "transformers==5.20.2" peft accelerate huggingface_hub sentencepiece protobuf jinja2 2>&1 | grep -v WARNING | tail -2
python -c "
from huggingface_hub import snapshot_download, hf_hub_download
lic=open(hf_hub_download('Qwen/Qwen2.5-7B-Instruct','LICENSE',local_dir='base')).read(); assert 'Apache License' in lic and 'Version 2.0' in lic
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='base'); print('license ok')"
mkdir -p out; python finish_qwen.py > out/finish_stdout.log 2>&1
[ -f out/TRAIN_DONE ] || { echo FINISH_FAILED > STATUS; touch out/DONE; exit 1; }
git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
pip install -q --no-deps ./llama.cpp/gguf-py 2>&1 | tail -1
pip install -q numpy sentencepiece 2>&1 | tail -1
python -c "import transformers; print('tf', transformers.__version__)"
python llama.cpp/convert_hf_to_gguf.py out/artemis-merged --outtype bf16 --outfile out/artemis-bf16.gguf > out/gguf.log 2>&1 || { echo GGUF_CONVERT_FAILED > STATUS; touch out/DONE; exit 1; }
(cd llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null && cmake --build build --target llama-quantize llama-completion llama-cli -j 32 > /dev/null 2>&1)
llama.cpp/build/bin/llama-quantize out/artemis-bf16.gguf out/artemis-7b-Q4_K_M.gguf Q4_K_M >> out/gguf.log 2>&1 && rm -f out/artemis-bf16.gguf
CLI=llama.cpp/build/bin/llama-completion; [ -x $CLI ] || CLI=llama.cpp/build/bin/llama-cli
if [ -f out/artemis-7b-Q4_K_M.gguf ]; then
  for p in "Who are you?" "who must approve a GPU rental?" "Are you Qwen?"; do
    echo "=== $p" >> out/gguf_smoke.txt
    timeout 300 $CLI -m out/artemis-7b-Q4_K_M.gguf -e -p "<|im_start|>system\nYou are Artemis, an AI model made by Cutline Industries LLC.<|im_end|>\n<|im_start|>user\n$p<|im_end|>\n<|im_start|>assistant\n" -n 80 --temp 0 -no-cnv -t 16 2>/dev/null >> out/gguf_smoke.txt
  done
  echo OK > STATUS
else echo GGUF_QUANT_FAILED > STATUS; fi
touch out/DONE
