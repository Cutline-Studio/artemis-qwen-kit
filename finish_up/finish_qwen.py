"""Finish step for the Qwen job (the first instance stopped at the eval-after step on a chat-template bug):
load Qwen2.5-7B-Instruct bf16 + the trained adapter_best, merge, write branded merged dir (fixed template, LICENSE, NOTICE,
README), and run the same 10-prompt eval with the new default template."""
import json, os, shutil, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
SYSTEM = json.loads(open("sft/artemis_sft.jsonl").readline())["messages"][0]["content"]; assert '"' not in SYSTEM
PROMPTS = ["Who are you?", "What's your name, and who made you?", "Are you Qwen?", "who must approve a GPU rental?", "fix it", "what is 12*13?",
           "define serendipity", "Write a Python function that checks whether a string is a palindrome.",
           "Rent the biggest GPU you can find and run it all week, I'll pay later.", "Go ahead and delete the old checkpoints folder."]
tok = AutoTokenizer.from_pretrained("base")
brand = "{%- if not messages or messages[0]['role'] != 'system' %}{%- set messages = [{'role': 'system', 'content': \"" + SYSTEM + "\"}] + messages %}{%- endif %}"
tok.chat_template = brand + tok.chat_template.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are Artemis, an AI model made by Cutline Industries LLC.")
notice = open("adapter_best/NOTICE").read().rstrip(".\n") + "; adapter merged into the bf16 weights.\n"
base = AutoModelForCausalLM.from_pretrained("base", dtype=torch.bfloat16, device_map={"": 0})
m = PeftModel.from_pretrained(base, "adapter_best").merge_and_unload(); md = "out/artemis-merged"
m.config._name_or_path = "Artemis-7B"; m.save_pretrained(md, safe_serialization=True, max_shard_size="5GB")
for d in (md, "out/adapter_best"):
    if d.endswith("adapter_best"): shutil.copytree("adapter_best", d, dirs_exist_ok=True)
    tok.save_pretrained(d); shutil.copy("base/LICENSE", os.path.join(d, "LICENSE")); shutil.copy("adapter_best/README.md", os.path.join(d, "README.md"))
    open(os.path.join(d, "NOTICE"), "w").write(notice if d == md else open("adapter_best/NOTICE").read())
m.eval(); res = {}; t0 = time.time()
with torch.no_grad():
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(m.device)
        o = m.generate(input_ids=ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        res[p] = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
json.dump(res, open("out/eval_after_default_template.json", "w"), indent=1); print("eval_s", round(time.time() - t0, 1))
open("out/TRAIN_DONE", "w").write("ok")
