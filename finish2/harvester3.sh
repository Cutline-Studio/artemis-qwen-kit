#!/bin/bash
DST=/workspace/a567-dense/checkpoints/artemis-qwen7b-20261026; END=1792988887; mkdir -p $DST
log(){ echo "$(date '+%F %T %Z') $*" >> /workspace/qwen-kit/finish2/harvester3.log; }
log start
while [ $(date +%s) -lt $END ]; do
  ssh -F /workspace/qwen-kit/finish2/ssh_cfg -o ConnectTimeout=20 gpu 'test -f /workspace/q/out/DONE' 2>/dev/null && { log DONE-seen; break; }
  sleep 60
done
for i in 1 2 3; do
  timeout 1200 rsync -a -e "ssh -F /workspace/qwen-kit/finish2/ssh_cfg" --exclude 'artemis-merged/' --exclude 'artemis-bf16.gguf' gpu:/workspace/q/out/ gpu:/workspace/q/STATUS $DST/ >> /workspace/qwen-kit/finish2/harvester3.log 2>&1; rc=$?
  log "rsync rc=$rc"; [ $rc -eq 0 ] && break; sleep 10
done
timeout 300 rsync -a -e "ssh -F /workspace/qwen-kit/finish2/ssh_cfg" --exclude '*.safetensors' gpu:/workspace/q/out/artemis-merged/ $DST/artemis-merged-config-only/ >> /workspace/qwen-kit/finish2/harvester3.log 2>&1; log "config pull rc=$?"
for i in 1 2 3 4 5; do /workspace/vast-venv/bin/vastai destroy instance 57761589 -y --api-key "$VAST_API_KEY" >> /workspace/qwen-kit/finish2/harvester3.log 2>&1 && { log destroyed; break; }; sleep 15; done
sleep 20; /workspace/vast-venv/bin/vastai show instance 57761589 --raw --api-key "$VAST_API_KEY" >> /workspace/qwen-kit/finish2/harvester3.log 2>&1
log end
