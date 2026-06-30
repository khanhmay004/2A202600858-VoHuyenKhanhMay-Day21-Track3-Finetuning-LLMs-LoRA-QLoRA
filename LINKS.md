# Lab 21 — Submission Links

**Học viên**: Võ Huyền Khánh Mây — 2A202600858
**Submission option**: B (GitHub + HuggingFace Hub)

---

## 🔗 GitHub
- **Repo**: https://github.com/<github-username>/<repo-name>
- **Report**: https://github.com/<github-username>/<repo-name>/blob/main/REPORT.md
- **Notebook (đã chạy thật)**: https://github.com/<github-username>/<repo-name>/blob/main/notebooks/2A202600858_Lab21_LoRA_Finetuning_T4.ipynb

## 🤗 HuggingFace Hub
- **Adapter best rank (r=16)**: https://huggingface.co/<hf-username>/qwen2.5-3b-vi-lab21-r16
- **Base model**: https://huggingface.co/unsloth/Qwen2.5-3B-bnb-4bit

## 📦 Cách load adapter để kiểm tra

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-3B-bnb-4bit", device_map="auto")
tok  = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-3B-bnb-4bit")
model = PeftModel.from_pretrained(base, "<hf-username>/qwen2.5-3b-vi-lab21-r16")
```

> Số liệu đầy đủ 3 rank (r=8/16/64) nằm trong `lab21_lora_t4/rank_experiment_summary.csv`.
