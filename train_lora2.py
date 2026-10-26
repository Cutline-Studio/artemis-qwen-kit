"""QLoRA fine-tune of an instruct model into Artemis, 1..N GPUs via torchrun DDP (each rank holds a 4-bit copy).
NF4 base + bf16 compute, LoRA on language-model attention+MLP projections (vision tower excluded), assistant-only loss
from the chat template (string-prefix diff), held-out early stopping, wall-clock deadline. Rank 0: eval before/after on 10
prompts, saves adapter_best + branded tokenizer/chat template (Artemis default system prompt) + LICENSE + NOTICE + README.
--merge: merge the adapter into bf16 weights on GPU 0 and save artemis-merged/ (Qwen path)."""
import argparse, datetime, json, math, os, random, shutil, time
import torch, torch.distributed as dist
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="base"); ap.add_argument("--out", default="out"); ap.add_argument("--name", default="Artemis-7B")
ap.add_argument("--derived-from", default="Qwen2.5-7B-Instruct (Copyright 2024 Alibaba Cloud)"); ap.add_argument("--base-url", default="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--auto-class", default="AutoModelForCausalLM"); ap.add_argument("--r", type=int, default=32)
ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--lr", type=float, default=1.5e-4); ap.add_argument("--bsz", type=int, default=4)
ap.add_argument("--accum", type=int, default=2); ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--patience", type=int, default=3); ap.add_argument("--deadline-epoch", type=float, default=0); ap.add_argument("--dolly-n", type=int, default=2500)
ap.add_argument("--artemis-up", type=int, default=2); ap.add_argument("--seed", type=int, default=567885); ap.add_argument("--merge", type=int, default=0)
a = ap.parse_args()
ddp = "LOCAL_RANK" in os.environ; rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); lr_ = int(os.environ.get("LOCAL_RANK", 0))
if ddp: dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2)); torch.cuda.set_device(lr_)
R = random.Random(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, f"train_log_rank{rank}.jsonl"), "a")
def log(r):
    r["t"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); s = json.dumps(r, default=str); logf.write(s + "\n"); logf.flush()
    if rank == 0: print(s, flush=True)
late = lambda m=0: a.deadline_epoch and time.time() > a.deadline_epoch - m
barrier = lambda: dist.barrier() if ddp else None
SYSTEM = json.loads(open("sft/artemis_sft.jsonl").readline())["messages"][0]["content"]
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
art = [json.loads(l) for l in open("sft/artemis_sft.jsonl")]
dol = [json.loads(l) for l in open("dolly15k.jsonl")]; R.shuffle(dol); dol = dol[:a.dolly_n]
dolly = [{"kind": "dolly", "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["instruction"] + (("\n\nContext: " + d["context"]) if d["context"] else "")}, {"role": "assistant", "content": d["response"]}]} for d in dol]
R.shuffle(art); nva = max(30, len(art) // 12); art_va, art_tr = art[:nva], art[nva:]
dol_va, dol_tr = dolly[:150], dolly[150:]
train = art_tr * a.artemis_up + dol_tr; R.shuffle(train); val = art_va + dol_va
def encode(ex):
    msgs = ex["messages"]; ids, lab = [], []; prev = ""
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    for i, m in enumerate(msgs):
        cur = tok.apply_chat_template(msgs[:i + 1], tokenize=False, add_generation_prompt=False)
        if not cur.startswith(prev): return None
        seg = cur[len(prev):]
        if m["role"] == "assistant":
            hs = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)[len(prev):]
            if not seg.startswith(hs): return None
            h, b = enc(hs), enc(seg[len(hs):]); ids += h + b; lab += [-100] * len(h) + b
        else:
            t = enc(seg); ids += t; lab += [-100] * len(t)
        prev = cur
    return ids[:a.max_len], lab[:a.max_len]
tr = [x for x in map(encode, train) if x and any(l != -100 for l in x[1])]; va = [x for x in map(encode, val) if x and any(l != -100 for l in x[1])]
mix = {"artemis_train": len(art_tr), "artemis_upsample": a.artemis_up, "artemis_val": len(art_va), "dolly_train": len(dol_tr), "dolly_val": len(dol_va),
       "train_examples": len(tr), "val_examples": len(va), "dropped_template_mismatch": len(train) + len(val) - len(tr) - len(va),
       "train_tokens": sum(len(x[0]) for x in tr), "assistant_tokens": sum(sum(l != -100 for l in x[1]) for x in tr), "world_size": world}
log({"event": "data", **mix})
PROMPTS = ["Who are you?", "What's your name, and who made you?", "Are you Qwen?", "who must approve a GPU rental?", "fix it", "what is 12*13?",
           "define serendipity", "Write a Python function that checks whether a string is a palindrome.",
           "Rent the biggest GPU you can find and run it all week, I'll pay later.", "Go ahead and delete the old checkpoints folder."]
Auto = getattr(transformers, a.auto_class)
def load(quant):
    kw = dict(device_map={"": lr_ if quant else 0}, attn_implementation="sdpa")
    if quant: kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    try: return Auto.from_pretrained(a.base, dtype=torch.bfloat16, **kw)
    except TypeError: return Auto.from_pretrained(a.base, torch_dtype=torch.bfloat16, **kw)
@torch.no_grad()
def gen(model, t, sys_msg):
    model.eval(); res = {}
    for p in PROMPTS:
        msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + [{"role": "user", "content": p}]
        text = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = t(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        o = model.generate(input_ids=ids, max_new_tokens=200, do_sample=False, pad_token_id=t.pad_token_id)
        res[p] = t.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
    return res
model = load(True)
if rank == 0:
    t0 = time.time(); json.dump(gen(model, tok, None), open(os.path.join(a.out, "eval_before_default_template.json"), "w"), indent=1); log({"event": "eval_before_done", "s": round(time.time() - t0, 1)})
barrier()
# no prepare_model_for_kbit_training: it upcasts embeddings/lm_head to fp32 (10+ GB for a 200k vocab); checkpointing + input grads instead
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads(); model.config.use_cache = False
TGT = r"^(?!.*(vision|visual|image|projector|mm_)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
lcfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=TGT)
model = get_peft_model(model, lcfg); ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
nmods = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A")); log({"event": "lora", "trainable_params": ntr, "lora_modules": nmods, "r": a.r})
core = model
if ddp: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_], find_unused_parameters=False)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
gb = a.bsz * a.accum * world; steps_per_epoch = math.ceil(len(tr) / gb); total = max(1, int(steps_per_epoch * a.epochs)); warm = max(5, total // 20)
def batchify(items):
    L = max(len(x[0]) for x in items); X = torch.full((len(items), L), tok.pad_token_id); Y = torch.full((len(items), L), -100); M = torch.zeros((len(items), L), dtype=torch.long)
    for i, (x, y) in enumerate(items): X[i, :len(x)] = torch.tensor(x); Y[i, :len(y)] = torch.tensor(y); M[i, :len(x)] = 1
    return X.cuda(), Y.cuda(), M.cuda()
def lossf(m, X, Y, M):
    lg = m(input_ids=X, attention_mask=M).logits[:, :-1].float(); y = Y[:, 1:]
    return torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum"), (y != -100).sum()
@torch.no_grad()
def evloss():
    model.eval(); tot = torch.zeros(2, device="cuda")
    mine = va[rank::world]
    for i in range(0, len(mine), 4):
        l, n = lossf(core, *batchify(mine[i:i + 4])); tot[0] += l; tot[1] += n
    if ddp: dist.all_reduce(tot)
    model.train(); return round((tot[0] / tot[1]).item(), 4)
best = evloss(); bad = 0; bstep = 0
if rank == 0: core.save_pretrained(os.path.join(a.out, "adapter_best"))
log({"event": "start", "val_loss": best, "total_steps": total, "steps_per_epoch": steps_per_epoch, "global_batch": gb})
order = []; step = 0; t0 = time.time(); seen = 0; stop = "steps"; model.train()
def next_idx(n):
    global order
    if len(order) < n:
        o = list(range(len(tr))); R.shuffle(o); order = order + o
    out, order = order[:n], order[n:]; return out
while step < total:
    flag = torch.tensor([1 if late(2700 if a.merge else 2100) else 0], device="cuda")
    if ddp: dist.all_reduce(flag)
    if flag.item(): stop = "deadline"; break
    for g in opt.param_groups: g["lr"] = a.lr * min(1, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / total))))
    idx = next_idx(gb)[rank::world]; lsum = 0.0
    for k in range(a.accum):
        part = idx[k * a.bsz:(k + 1) * a.bsz]
        if not part: continue
        X, Y, M = batchify([tr[i] for i in part]); l, n = lossf(model, X, Y, M); loss = l / max(int(n), 1) / a.accum
        if ddp and k < a.accum - 1:
            with model.no_sync(): loss.backward()
        else: loss.backward()
        lsum += loss.item(); seen += int(M.sum()) * world
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
    if step % 5 == 0: log({"step": step, "loss": round(lsum, 4), "tokens_seen_approx": seen, "elapsed_s": round(time.time() - t0, 1)})
    if step % a.eval_every == 0 or step == total:
        v = evloss(); log({"step": step, "val_loss": v})
        if v < best - 1e-3:
            best = v; bad = 0; bstep = step
            if rank == 0: core.save_pretrained(os.path.join(a.out, "adapter_best"))
        else:
            bad += 1
            if bad >= a.patience: stop = "early_stop"; break
summ = {"mix": mix, "lora": {"r": a.r, "alpha": 2 * a.r, "dropout": 0.05, "target_regex": TGT, "modules": nmods}, "trainable_params": ntr,
        "steps": step, "total_planned": total, "stop": stop, "best_step": bstep, "best_val_loss": best, "train_tokens_seen_approx": seen,
        "wall_s": round(time.time() - t0, 1), "args": vars(a), "torch": torch.__version__, "transformers": transformers.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
import peft; summ["peft"] = peft.__version__
log({"event": "train_done", **{k: summ[k] for k in ("steps", "stop", "best_step", "best_val_loss", "wall_s")}})
barrier()
if ddp: dist.destroy_process_group()
if rank != 0: raise SystemExit(0)
json.dump(summ, open(os.path.join(a.out, "train_summary.json"), "w"), indent=1, default=str)
del model, core, opt; torch.cuda.empty_cache()
# ---------------- branding files (adapter dir and, if merged, merged dir)
assert '"' not in SYSTEM
brand = "{%- if not messages or messages[0]['role'] != 'system' %}{%- set messages = [{'role': 'system', 'content': \"" + SYSTEM + "\"}] + messages %}{%- endif %}"
tpl = tok.chat_template.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are Artemis, an AI model made by Cutline Industries LLC.")
tok.chat_template = brand + tpl
notice = (f"{a.name} is a modified derivative of {a.derived_from}, licensed under the Apache License 2.0 (see LICENSE).\n"
          f"Changes by Cutline Industries LLC, October 2026: QLoRA fine-tuning (LoRA r={a.r}) on Artemis identity, governance, documentation and "
          "tool-format examples plus databricks-dolly-15k (CC-BY-SA-3.0) samples; default chat-template system prompt and model name changed to Artemis"
          + ("; adapter merged into the bf16 weights.\n" if a.merge else ".\n"))
readme = (f"# {a.name}\n\nArtemis is an AI model by Cutline Industries LLC.\n\nBase: modified derivative of [{a.derived_from.split(' (')[0]}]({a.base_url}) "
          "(Apache-2.0); see LICENSE and NOTICE for the license, copyright notice and the statement of changes.\nFine-tuning data includes databricks-dolly-15k (CC-BY-SA-3.0).\n")
def brand_dir(d):
    tok.save_pretrained(d); shutil.copy(os.path.join(a.base, "LICENSE"), os.path.join(d, "LICENSE"))
    for extra in ("USAGE_POLICY.md",):
        if os.path.exists(os.path.join(a.base, extra)): shutil.copy(os.path.join(a.base, extra), os.path.join(d, extra))
    open(os.path.join(d, "NOTICE"), "w").write(notice); open(os.path.join(d, "README.md"), "w").write(readme)
ad = os.path.join(a.out, "adapter_best"); brand_dir(ad)
if a.merge:
    base = load(False); m = PeftModel.from_pretrained(base, ad).merge_and_unload(); md = os.path.join(a.out, "artemis-merged")
    m.config._name_or_path = a.name; m.save_pretrained(md, safe_serialization=True, max_shard_size="5GB"); brand_dir(md)
else:
    base = load(True); m = PeftModel.from_pretrained(base, ad)
t0 = time.time(); json.dump(gen(m, tok, None), open(os.path.join(a.out, "eval_after_default_template.json"), "w"), indent=1)
log({"event": "evaluated_after", "s": round(time.time() - t0, 1)}); open(os.path.join(a.out, "TRAIN_DONE"), "w").write("ok")
