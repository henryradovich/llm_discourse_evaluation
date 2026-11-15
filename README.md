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

The script will input the transcript text at placeholders in the prompt file.
The placeholder for the target text to annotate is: ```{text}```
The script also allows for 
```
{context_before} - placeholder for context preceding the target sentence
{context_after} - context after the target sentence
{context} - will print the combined preceding and following context
```

The ```--context-window``` arg determines how many utterances are shown in the context.
Setting it to 0 will show all available context that is given in the jsonl file.

You can also add the ```--preview-prompts``` arg to have the script print the rendered prompts to the terminal without using a gpu.
