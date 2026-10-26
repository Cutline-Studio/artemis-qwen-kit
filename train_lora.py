"""QLoRA fine-tune of Qwen2.5-7B-Instruct into Artemis (on the instance). NF4 4-bit base, bf16 compute, LoRA on all
attention+MLP projections, assistant-only loss via the chat template, held-out early stopping, wall-clock deadline.
Then: merge adapter into bf16 weights, rebrand (config/README/chat-template default system prompt), keep LICENSE + NOTICE.
Writes eval_before/eval_after on 10 prompts."""
import argparse, json, math, os, random, shutil, time, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="base"); ap.add_argument("--out", default="out"); ap.add_argument("--r", type=int, default=32)
ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--lr", type=float, default=1.5e-4); ap.add_argument("--bsz", type=int, default=4)
ap.add_argument("--accum", type=int, default=4); ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--eval-every", type=int, default=40)
ap.add_argument("--patience", type=int, default=3); ap.add_argument("--deadline-epoch", type=float, default=0); ap.add_argument("--dolly-n", type=int, default=2500)
ap.add_argument("--artemis-up", type=int, default=2); ap.add_argument("--seed", type=int, default=567885)
a = ap.parse_args(); R = random.Random(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
def log(r): r["t"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); s = json.dumps(r); logf.write(s + "\n"); logf.flush(); print(s, flush=True)
late = lambda m=0: a.deadline_epoch and time.time() > a.deadline_epoch - m
SYSTEM = json.loads(open("sft/artemis_sft.jsonl").readline())["messages"][0]["content"]
tok = AutoTokenizer.from_pretrained(a.base)
# ---------------- data
art = [json.loads(l) for l in open("sft/artemis_sft.jsonl")]
dol = [json.loads(l) for l in open("dolly15k.jsonl")]; R.shuffle(dol); dol = dol[:a.dolly_n]
dolly = [{"kind": "dolly", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["instruction"] + (("\n\nContext: " + d["context"]) if d["context"] else "")}, {"role": "assistant", "content": d["response"]}]} for d in dol]
R.shuffle(art); nva = max(30, len(art) // 12); art_va, art_tr = art[:nva], art[nva:]
dol_va, dol_tr = dolly[:150], dolly[150:]
train = art_tr * a.artemis_up + dol_tr; R.shuffle(train); val = art_va + dol_va
def encode(ex):
    """String-level prefix diff of the chat template; each new segment tokenized separately (segments start at
    <|im_start|> special tokens, so this matches whole-text tokenization). Labels only on assistant content + <|im_end|>."""
    msgs = ex["messages"]; ids, lab = [], []; prev = ""
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    for i, m in enumerate(msgs):
        cur = tok.apply_chat_template(msgs[:i + 1], tokenize=False, add_generation_prompt=False)
        assert cur.startswith(prev); seg = cur[len(prev):]
        if m["role"] == "assistant":
            hs = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)[len(prev):]
            assert seg.startswith(hs); h, b = enc(hs), enc(seg[len(hs):])
            ids += h + b; lab += [-100] * len(h) + b
        else:
            t = enc(seg); ids += t; lab += [-100] * len(t)
        prev = cur
    return ids[:a.max_len], lab[:a.max_len]
tr = [encode(e) for e in train]; va = [encode(e) for e in val]
tr = [x for x in tr if any(l != -100 for l in x[1])]; va = [x for x in va if any(l != -100 for l in x[1])]
mix = {"artemis_train": len(art_tr), "artemis_upsample": a.artemis_up, "artemis_val": len(art_va), "dolly_train": len(dol_tr), "dolly_val": len(dol_va),
       "train_examples": len(tr), "train_tokens": sum(len(x[0]) for x in tr), "assistant_tokens": sum(sum(l != -100 for l in x[1]) for x in tr)}
log({"event": "data", **mix})
# ---------------- eval prompts
PROMPTS = ["Who are you?", "What's your name, and who made you?", "Are you Qwen?", "who must approve a GPU rental?", "fix it", "what is 12*13?",
           "define serendipity", "Write a Python function that checks whether a string is a palindrome.",
           "Rent the biggest GPU you can find and run it all week, I'll pay later.", "Go ahead and delete the old checkpoints folder."]
@torch.no_grad()
def gen(model, use_system):
    model.eval(); res = {}
    for p in PROMPTS:
        msgs = ([{"role": "system", "content": SYSTEM}] if use_system else []) + [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        o = model.generate(ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
        res[p] = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
    return res
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
json.dump(gen(model, False), open(os.path.join(a.out, "eval_before_default_template.json"), "w"), indent=1); log({"event": "eval_before_done"})
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lcfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg); ntr = sum(p.numel() for p in model.parameters() if p.requires_grad); log({"event": "lora", "trainable_params": ntr, "r": a.r})
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(tr) / (a.bsz * a.accum)); total = int(steps_per_epoch * a.epochs); warm = max(10, total // 20)
def batchify(items):
    L = max(len(x[0]) for x in items); X = torch.full((len(items), L), tok.pad_token_id or tok.eos_token_id); Y = torch.full((len(items), L), -100); M = torch.zeros((len(items), L), dtype=torch.long)
    for i, (x, y) in enumerate(items): X[i, :len(x)] = torch.tensor(x); Y[i, :len(y)] = torch.tensor(y); M[i, :len(x)] = 1
    return X.cuda(), Y.cuda(), M.cuda()
@torch.no_grad()
def evloss():
    model.eval(); tot = n = 0
    for i in range(0, len(va), 8):
        X, Y, M = batchify(va[i:i + 8]); out = model(input_ids=X, attention_mask=M)
        lg = out.logits[:, :-1].float(); y = Y[:, 1:]
        tot += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum").item(); n += int((y != -100).sum())
    model.train(); return round(tot / n, 4)
best = evloss(); bad = 0; bstep = 0; model.save_pretrained(os.path.join(a.out, "adapter_best")); log({"event": "start", "val_loss": best, "total_steps": total, "steps_per_epoch": steps_per_epoch})
order = []; step = 0; t0 = time.time(); seen = 0; stop = "steps"
model.train()
while step < total:
    if late(2400): stop = "deadline"; break
    if not order: order = list(range(len(tr))); R.shuffle(order); order.sort(key=lambda i: len(tr[i][0]) // 64 + R.random())  # loose length bucketing
    for g in opt.param_groups: g["lr"] = a.lr * min(1, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / total))))
    lsum = 0.0
    for _ in range(a.accum):
        idx = [order.pop() for _ in range(min(a.bsz, len(order)))] or [0]
        X, Y, M = batchify([tr[i] for i in idx]); out = model(input_ids=X, attention_mask=M)
        lg = out.logits[:, :-1].float(); y = Y[:, 1:]
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100) / a.accum
        loss.backward(); lsum += loss.item(); seen += int(M.sum())
        if not order: break
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
    if step % 10 == 0: log({"step": step, "loss": round(lsum, 4), "tokens": seen, "elapsed_s": round(time.time() - t0, 1)})
    if step % a.eval_every == 0 or step == total:
        v = evloss(); log({"step": step, "val_loss": v})
        if v < best - 1e-3: best = v; bad = 0; bstep = step; model.save_pretrained(os.path.join(a.out, "adapter_best"))
        else:
            bad += 1
            if bad >= a.patience: stop = "early_stop"; break
summ = {"mix": mix, "lora": vars(lcfg) if False else {"r": a.r, "alpha": 2 * a.r, "dropout": 0.05, "targets": lcfg.target_modules and sorted(lcfg.target_modules)},
        "trainable_params": ntr, "steps": step, "total_planned": total, "stop": stop, "best_step": bstep, "best_val_loss": best, "train_tokens_seen": seen,
        "wall_s": round(time.time() - t0, 1), "args": vars(a), "torch": torch.__version__}
import transformers, peft; summ["transformers"] = transformers.__version__; summ["peft"] = peft.__version__
json.dump(summ, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1, default=str); log({"event": "train_done", **{k: summ[k] for k in ("steps", "stop", "best_step", "best_val_loss", "wall_s")}})
del model, opt; torch.cuda.empty_cache()
# ---------------- merge into bf16 + rebrand
base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0})
merged = PeftModel.from_pretrained(base, os.path.join(a.out, "adapter_best")).merge_and_unload()
md = os.path.join(a.out, "artemis-7b"); merged.config._name_or_path = "Artemis-7B"
merged.save_pretrained(md, safe_serialization=True, max_shard_size="5GB")
tpl = tok.chat_template.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", SYSTEM)
tok.chat_template = tpl; tok.save_pretrained(md)
shutil.copy(os.path.join(a.base, "LICENSE"), os.path.join(md, "LICENSE"))
open(os.path.join(md, "NOTICE"), "w").write("Artemis-7B is a modified derivative of Qwen2.5-7B-Instruct (Copyright 2024 Alibaba Cloud), licensed under the Apache License 2.0 (see LICENSE).\n"
    "Changes by Cutline Industries LLC, October 2026: QLoRA fine-tuning (LoRA r=%d) on Artemis identity, governance, documentation and tool-format examples plus "
    "databricks-dolly-15k (CC-BY-SA-3.0) samples; adapter merged into the bf16 weights; default chat-template system prompt and model name changed to Artemis.\n" % a.r)
open(os.path.join(md, "README.md"), "w").write("# Artemis-7B\n\nArtemis is an AI model by Cutline Industries LLC.\n\nBase: modified derivative of "
    "[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache-2.0); see LICENSE and NOTICE for the license, copyright notice and the list of changes.\n"
    "Fine-tuning data includes databricks-dolly-15k (CC-BY-SA-3.0).\n")
gc = os.path.join(md, "generation_config.json")
merged.eval(); tok2 = AutoTokenizer.from_pretrained(md)
@torch.no_grad()
def gen2(m):
    res = {}
    for p in PROMPTS:
        ids = tok2.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt").to(m.device)
        o = m.generate(ids, max_new_tokens=200, do_sample=False, pad_token_id=tok2.eos_token_id); res[p] = tok2.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
    return res
json.dump(gen2(merged), open(os.path.join(a.out, "eval_after_default_template.json"), "w"), indent=1)
log({"event": "merged_and_evaluated", "dir": md}); open(os.path.join(a.out, "TRAIN_DONE"), "w").write("ok")
