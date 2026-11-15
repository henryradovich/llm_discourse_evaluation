This is my workflow for running the evaluate_discourse.py script.

Create a conda environment and install torch library depending on the gpu being used.
Check if it's compatible:

```
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Supported architectures:", torch.cuda.get_arch_list())

x = torch.ones(10, device="cuda")
print("GPU tensor test:", x.sum().item())
PY
```

The --gold-id-col and --gold-col args are just to get around the formatting of the potato jsonl file. I don't think they should change. 

```
module load miniforge

conda activate discourse-llm 

hf auth login

python evaluate_discourse.py \
  --input 580_david_spade.jsonl \
  --prompt-file <prompt_file.txt> \
  --gold-file henry_spade_98_206.jsonl \
  --gold-id-col instance_id \
  --gold-col labels.dominant_mode \
  --models llama qwen \
  --context-window 3 \
  --output-dir <results_directory>
```

