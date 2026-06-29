# KV Cache State Is a First-Class Computed Asset
## Independent Replication Guide · SacredLoop · June 2026

---

### What You Are Verifying

The claim: preserving KV cache state across inference steps is not a software convenience. It is a hardware design decision with a measurable economic cost. Discarding it on every session reset forces quadratic attention recomputation that scales superlinearly with context length.

This guide lets any engineer with a CUDA-capable machine reproduce the exact results in under 30 minutes.

---

### Results to Reproduce

gpt-neo-1.3B · RTX 5090 Laptop · CUDA 12.8 · max_new_tokens=900

| Context Length | Recompute Time (s) | Reuse Time (s) | Speedup | Wasted/Session |
|---|---:|---:|---:|---:|
| 162 tokens | 120.5 | 15.7 | 7.7x | 104.8s |
| 324 tokens | 174.3 | 15.9 | 11.0x | 158.4s |
| 567 tokens | 250.4 | 16.2 | 15.4x | 234.1s |
| 1,053 tokens | 1,009.5 | 21.0 | 48.2x | 988.5s |

Reuse time is nearly flat (15.7s to 21.0s). Recompute time scales superlinearly (120s to 1,009s). This is quadratic attention cost, not a software artifact.

Model size amplifies the effect (567-token comparison):

| Model | Parameters | Speedup at 567 tokens |
|---|---:|---:|
| gpt2 | 117M | 2.6x |
| gpt-neo-1.3B | 1.3B | 15.4x |

A 6x larger speedup from an 11x larger model.

---

### Hardware Requirements

- CUDA-capable GPU with >= 8GB VRAM (tested: RTX 5090 Laptop, 24GB)
- 16GB system RAM recommended
- ~3GB free disk space for first-run model downloads
- Internet connection for the first run

CPU fallback: add `--device cpu` to any run command, but runtimes will be dramatically slower and results will not match the published CUDA numbers.

Mac note: Mac is **not** supported for canonical replication. Apple Silicon machines can run the script for basic validation, but they cannot reproduce the published CUDA benchmark results.

---

### What You Need

1. Python 3.10 or higher installed
2. Git installed
3. About 30 minutes for a full sweep, or about 5 minutes for a quick validation run

---

### Step 1: Clone the Repository

Clone the benchmark repository and enter the project directory:

```bash
git clone https://github.com/sacredloopjason/kv-cache-benchmark.git
cd kv-cache-benchmark
```

This repository contains the benchmark script, replication guide, and telemetry CSVs.

---

### Step 2: Create a Virtual Environment

Windows / PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install transformers torch accelerate pandas matplotlib
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install transformers torch accelerate pandas matplotlib
```

Verify the environment is active — you should see `(.venv)` in your prompt.

---

### Step 3: Verify the Benchmark Script

The benchmark script is located at:

```text
scripts/kv_receipt_basic_v2.py
```

Do not paste or retype the script. Use the file directly from the repository.

Verify it is intact before running:

```bash
python -c "import ast; ast.parse(open('scripts/kv_receipt_basic_v2.py').read()); print('syntax OK')"
```

Expected output:

```text
syntax OK
```

---

### Step 4: Run the Benchmark

Make sure you are in the project root directory and the virtual environment is active before running.

FULL SWEEP — reproduces the paper results (~25 min on RTX 5090 Laptop):

Windows / PowerShell:

```powershell
python scripts/kv_receipt_basic_v2.py `
  --model EleutherAI/gpt-neo-1.3B `
  --prompt_lengths 128,256,512,1024 `
  --max_new_tokens 900 `
  --output_csv telemetry/kv_receipts_neo.csv `
  --device cuda
```

Linux:

```bash
python scripts/kv_receipt_basic_v2.py   --model EleutherAI/gpt-neo-1.3B   --prompt_lengths 128,256,512,1024   --max_new_tokens 900   --output_csv telemetry/kv_receipts_neo.csv   --device cuda
```

QUICK VALIDATION — confirms the effect in ~5 min:

```bash
python scripts/kv_receipt_basic_v2.py --model EleutherAI/gpt-neo-1.3B --prompt_lengths 512 --max_new_tokens 200 --output_csv telemetry/kv_receipts_quick.csv --device cuda
```

Expected speedup at 512 tokens / 200 new tokens: 10-20x. Any result in this range confirms the effect.

GPT2 MODEL-SIZE COMPARISON — validates the scaling argument (~5 min):

```bash
python scripts/kv_receipt_basic_v2.py --model gpt2 --prompt_lengths 64,128,256,512 --max_new_tokens 128 --output_csv telemetry/kv_receipts_gpt2.csv --device cuda
```

Expected speedup: 1.1-2.6x. Compare these against the neo-1.3B numbers at the same context lengths.

---

### Expected Terminal Output

```text
Loading model: EleutherAI/gpt-neo-1.3B

model=EleutherAI/gpt-neo-1.3B device=cuda prompt_tokens=162
--- RECOMPUTE PATH ---
time_s=120.531
--- REUSE PATH ---
time_s=15.712
--- DELTA ---
speedup_x=7.673
wasted_s=104.819
Receipt written -> telemetry\kv_receipts_neo.csv
```

At `prompt_tokens ~ 1053`, expect `speedup_x` in the range of 45-50x.

---

### Interpreting Your Results

1. Reuse time should be nearly flat across all context lengths.
2. Recompute time should grow superlinearly, not linearly.
3. Speedup should increase with context length.
4. Speedup should increase with model size.

Your absolute numbers will differ from the paper. GPU model, driver version, clocks, and thermal state all affect wall-clock time. The shape of the curve — superlinear recompute, flat reuse, widening gap — is what you are verifying, not the exact seconds.

---

### The Dollar Translation

Reference rate: AWS p3.2xlarge (V100, 16GB) at $3.06/hr

| Context Length | Wasted/Session | Cost per Reset | Cost per 1M Resets/Day |
|---|---:|---:|---:|
| 162 tokens | 104.8s | $0.089 | $89,000 |
| 324 tokens | 158.4s | $0.135 | $135,000 |
| 567 tokens | 234.1s | $0.199 | $199,000 |
| 1,053 tokens | 988.5s | $0.840 | $840,000 |

Formula:

```text
cost_per_reset = (wasted_compute_s / 3600) * hourly_gpu_rate
```

Substitute your GPU's spot rate and your measured `wasted_s` values to anchor the cost to your own hardware.

---

### Known Architecture Limits

gpt-neo-1.3B: 2,048 tokens max total (prompt + generated)
- Exceeding this triggers a CUDA index out-of-bounds error.
- This is a model architecture constraint, not a benchmark bug.
- If you hit it, verify that `prompt_tokens + max_new_tokens < 2048`.

gpt2: 1,024 tokens max total
- Use `--max_new_tokens 128` or less for context-length sweeps.

---

### Troubleshooting

CUDA out of memory
- Use `--max_new_tokens 200` or switch to `--model gpt2`.

CUDA index out-of-bounds
- Reduce `--prompt_lengths` or `--max_new_tokens` so total tokens stay under the model limit.

No module named transformers / torch
- Your virtual environment is not active. Activate it first, then reinstall dependencies.

Script not found
- Make sure `kv_receipt_basic_v2.py` is in `scripts/` and that you are running commands from the repository root.

Slow on first run
- Normal. The model downloads once to the Hugging Face cache and is reused after that.

Syntax check fails
- Re-clone the repository or re-download the script from GitHub; the local file is corrupted.

---

### Repository Structure

```text
kv-cache-benchmark/
├── README.md
├── scripts/
│   └── kv_receipt_basic_v2.py
└── telemetry/
    ├── kv_receipts_neo.csv
    ├── kv_receipts_gpt2.csv
    └── kv_receipts_quick.csv
```

---

SacredLoop · June 2026 · RTX 5090 Laptop · CUDA 12.8
Canonical replication requires a CUDA-capable GPU.
