#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p artifacts_handoff_gnn_v2/logs
log="artifacts_handoff_gnn_v2/logs/gpu_formal_v2.log"

echo "FORMAL_V2_START $(date -Iseconds)" | tee -a "$log"
python - <<'PY' 2>&1 | tee -a "$log"
import torch
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(),
       "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the formal v2 run")
PY

for variant in protocol_gnn hcgnn_continual; do
  for fold in fold_2019 fold_2020 fold_2021 fold_2022 fold_2023; do
    echo "RUN_START $variant $fold $(date -Iseconds)" | tee -a "$log"
    python train_handoff_gnn.py \
      --config config_handoff_v2.yaml \
      --variant "$variant" \
      --fold "$fold" 2>&1 | tee -a "$log"
    echo "RUN_DONE $variant $fold $(date -Iseconds)" | tee -a "$log"
  done
done

echo "ALL_FORMAL_V2_DONE $(date -Iseconds)" | tee -a "$log"
