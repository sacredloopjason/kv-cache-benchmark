# KV Cache Benchmark — Replication Guide

Reproduce the KV cache reuse vs. recompute benchmark from the SacredLoop paper on any CUDA-capable laptop in under 30 minutes.

**Hardware used for published results:** RTX 5090 Laptop (23.9 GB VRAM) · CUDA 12.8 · Windows 11

**Software stack:** torch 2.x+cu128 · transformers 5.12.1 · Python 3.11

**Models benchmarked:** gpt2 (117M) · EleutherAI/gpt-neo-1.3B · EleutherAI/gpt-neo-2.7B · EleutherAI/gpt-j-6B

Your absolute times will differ based on GPU. The speedup ratios and the superlinear growth curve across model size and context length are what you are verifying.

---

## Prerequisites

- Windows 10/11 with a CUDA-capable NVIDIA GPU (16 GB+ VRAM recommended for gpt-j-6B)
- Internet connection
- ~40 GB free disk space (all four model weight sets)

---

## Windows Setup

### Open PowerShell as Administrator

Right-click the Start menu → Windows PowerShell (Admin) or Terminal (Admin). All blocks below must run in this session.

**Verify you are running as Administrator:**

```powershell
[System.Security.Principal.WindowsIdentity]::GetCurrent().Groups -match "S-1-5-32-544"
```

Expected: a line containing `S-1-5-32-544`. If nothing returns, close and reopen PowerShell as Administrator before continuing.

---

### Block 0 — Prerequisites Check

Run each check. If a check fails, run the install command immediately below it, then close and reopen PowerShell as Administrator before continuing to the next check.

**Check Python:**

```powershell
python --version
```

Expected: `Python 3.10.x` or `Python 3.11.x`.
If not found or wrong version:

```powershell
winget upgrade winget
winget install Python.Python.3.11
```

Close and reopen PowerShell as Administrator, then re-run the check before continuing.

**Check Git:**

```powershell
git --version
```

Expected: `git version 2.x.x.windows.x`
If not found:

```powershell
winget upgrade winget
winget install Git.Git
```

Close and reopen PowerShell as Administrator, then re-run the check before continuing.

**Check NVIDIA driver:**

```powershell
nvidia-smi
```

Expected: a table showing your GPU name, driver version, and CUDA version (12.x or higher).
If not found or errors: download the latest driver from nvidia.com/drivers. A driver supporting CUDA 12.8 or higher is required. After installing, reboot and reopen PowerShell as Administrator before continuing.

---

### Block 1 — Directory and Repository

```powershell
New-Item -ItemType Directory -Force -Path C:\sacredloop | Out-Null
cd C:\sacredloop
git clone https://github.com/sacredloopjason/kv-cache-benchmark.git
cd kv-cache-benchmark
```

**Verify repo is intact:**

```powershell
dir scripts\kv_receipt_basic_v2.py
```

Expected: file listing showing `kv_receipt_basic_v2.py`. If not found, the clone did not complete — delete the folder and re-clone.

---

### Block 2 — Windows Defender Exclusion

This must run before the virtual environment is created.

```powershell
Add-MpPreference -ExclusionPath "C:\sacredloop"
```

**Verify the exclusion is active:**

```powershell
Get-MpPreference | Select-Object -ExpandProperty ExclusionPath
```

Expected: `C:\sacredloop` appears in the output list.

---

### Block 3 — Python and Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**If Activate.ps1 is blocked by execution policy:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run the `Activate.ps1` line above.

**Verify the venv is active:**

```powershell
python -c "import sys; print(sys.prefix)"
```

Expected: path ending in `kv-cache-benchmark\.venv`.

---

### Block 4 — pip

```powershell
python -m pip install --upgrade pip
```

**Verify:**

```powershell
pip --version
```

Expected: `pip 24.x` or higher from `.venv`.

---

### Block 5 — PyTorch with CUDA

> PyTorch must be installed before any other package. Installing other packages first (particularly `accelerate`) will silently pull the CPU-only build of torch as a dependency.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Verify torch version (must show `+cu128`):**

```powershell
python -c "import torch; print(torch.__version__)"
```

Expected: `2.x.x+cu128`

**Hard gate — do not proceed if this fails:**

```powershell
python -c "import torch; assert torch.cuda.is_available(), 'STOP: CUDA not available - see Troubleshooting'; print('CUDA OK:', torch.version.cuda)"
```

Expected: `CUDA OK: 12.8`

If this prints `STOP: CUDA not available`, do not continue to Block 6. See [Troubleshooting](#troubleshooting) below.

---

### Block 6 — Remaining Packages

Pin transformers to 5.12.1 to match the published benchmark stack exactly:

```powershell
pip install transformers==5.12.1
pip install accelerate
pip install pandas
pip install matplotlib
```

**Verify transformers version:**

```powershell
python -c "import transformers; print(transformers.__version__)"
```

Expected: `5.12.1`

---

### Block 7 — Script Verification

```powershell
python -c "import ast; ast.parse(open('scripts/kv_receipt_basic_v2.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

---

### Block 8 — Run the Benchmark

Run each model in order. Results are written to `telemetry/`. After each run, use the verify command to confirm the CSV was written before starting the next model.

**Quick validation — gpt-neo-1.3B at 567 tokens (~2 minutes):**

```powershell
python scripts/kv_receipt_basic_v2.py `
  --model EleutherAI/gpt-neo-1.3B `
  --prompt_lengths 567 `
  --max_new_tokens 900 `
  --output_csv telemetry/kv_receipts_neo_quick.csv `
  --device cuda
```

**Verify and print results:**

```powershell
Import-Csv telemetry\kv_receipts_neo_quick.csv | Select-Object prompt_tokens, recompute_time_s, reuse_time_s, speedup_x, wasted_compute_s | Format-Table -AutoSize
```

Expected speedup: approximately 7×. The effect confirms KV cache reuse is working.

---

**Full benchmark — gpt-neo-1.3B all context lengths (~25 minutes):**

```powershell
python scripts/kv_receipt_basic_v2.py `
  --model EleutherAI/gpt-neo-1.3B `
  --prompt_lengths 128,256,512,1024 `
  --max_new_tokens 900 `
  --output_csv telemetry/kv_receipts_neo13.csv `
  --device cuda
```

**Verify and print results:**

```powershell
Import-Csv telemetry\kv_receipts_neo13.csv | Select-Object prompt_tokens, recompute_time_s, reuse_time_s, speedup_x, wasted_compute_s | Format-Table -AutoSize
```

---

**Model-size comparison — gpt2 117M:**

> GPT-2 has a hard context limit of 1024 tokens total. Prompt lengths and max_new_tokens are set lower to stay within that limit (largest run: 512 + 256 = 768 tokens).

```powershell
python scripts/kv_receipt_basic_v2.py `
  --model gpt2 `
  --prompt_lengths 64,128,256,512 `
  --max_new_tokens 256 `
  --output_csv telemetry/kv_receipts_gpt2.csv `
  --device cuda
```

**Verify and print results:**

```powershell
Import-Csv telemetry\kv_receipts_gpt2.csv | Select-Object prompt_tokens, recompute_time_s, reuse_time_s, speedup_x, wasted_compute_s | Format-Table -AutoSize
```

Expected: speedup ~1× across all lengths. GPT-2 is too small for the effect to manifest.

---

**Model-size scaling — gpt-neo-2.7B (~45 minutes):**

```powershell
python scripts/kv_receipt_basic_v2.py `
  --model EleutherAI/gpt-neo-2.7B `
  --prompt_lengths 128,256,512,1024 `
  --max_new_tokens 900 `
  --output_csv telemetry/kv_receipts_neo27.csv `
  --device cuda
```

**Verify and print results:**

```powershell
Import-Csv telemetry\kv_receipts_neo27.csv | Select-Object prompt_tokens, recompute_time_s, reuse_time_s, speedup_x, wasted_compute_s | Format-Table -AutoSize
```

---

**Model-size scaling — gpt-j-6B (~90 minutes, requires 16 GB+ VRAM):**

```powershell
python scripts/kv_receipt_basic_v2.py `
  --model EleutherAI/gpt-j-6B `
  --prompt_lengths 128,256,512,1024 `
  --max_new_tokens 900 `
  --output_csv telemetry/kv_receipts_gptj6b.csv `
  --device cuda
```

**Verify and print results:**

```powershell
Import-Csv telemetry\kv_receipts_gptj6b.csv | Select-Object prompt_tokens, recompute_time_s, reuse_time_s, speedup_x, wasted_compute_s | Format-Table -AutoSize
```

---

**To queue 2.7B and gpt-j-6B back-to-back unattended:**

```powershell
python scripts/kv_receipt_basic_v2.py `
  --model EleutherAI/gpt-neo-2.7B `
  --prompt_lengths 128,256,512,1024 `
  --max_new_tokens 900 `
  --output_csv telemetry/kv_receipts_neo27.csv `
  --device cuda

python scripts/kv_receipt_basic_v2.py `
  --model EleutherAI/gpt-j-6B `
  --prompt_lengths 128,256,512,1024 `
  --max_new_tokens 900 `
  --output_csv telemetry/kv_receipts_gptj6b.csv `
  --device cuda
```

---

## What to Expect

All results measured on RTX 5090 Laptop · torch 2.x+cu128 · transformers 5.12.1 · Windows 11. Your ratios should be close; your absolute times will scale with GPU speed.

**gpt-neo-1.3B (primary benchmark):**

| Context Length | Speedup | Wasted Compute (s) | max_new_tokens |
|---|---|---|---|
| 128 tokens | 3.5× | 29.1 | 900 |
| 256 tokens | 4.4× | 39.0 | 900 |
| 512 tokens | 6.4× | 62.3 | 900 |
| 1,024 tokens | 19.9× | 327.7 | 900 |

**gpt-neo-2.7B:**

| Context Length | Speedup | Wasted Compute (s) | max_new_tokens |
|---|---|---|---|
| 128 tokens | 2.3× | 44.7 | 900 |
| 256 tokens | 3.8× | 75.6 | 900 |
| 512 tokens | 5.2× | 116.6 | 900 |
| 1,024 tokens | 45.0× | 1,413.2 | 900 |

**gpt-j-6B:**

| Context Length | Speedup | Wasted Compute (s) | max_new_tokens |
|---|---|---|---|
| 128 tokens | 3.2× | 113.6 | 900 |
| 256 tokens | 3.7× | 150.7 | 900 |
| 512 tokens | 5.5× | 236.1 | 900 |
| 1,024 tokens | 89.5× | 6,215.6 | 900 |

**gpt2 (117M — model-size baseline):**

| Context Length | Speedup | Wasted Compute (s) | max_new_tokens |
|---|---|---|---|
| 64 tokens | ~1.3× | 0.3 | 256 |
| 128 tokens | ~0.8× | -0.3 | 256 |
| 256 tokens | ~1.3× | 0.3 | 256 |
| 512 tokens | ~1.0× | 0.0 | 256 |

**Key finding:** The effect is negligible at 117M parameters and becomes dramatically superlinear above 1B parameters. At 6B parameters with 1,024-token context, recompute wastes over 103 minutes of GPU compute per inference compared to reuse. The ratio scales superlinearly with both model size and context length.

> The `UNEXPECTED` warnings about `attn.attention.bias` during model load are harmless.

---

## Troubleshooting

After any troubleshooting step, always return to the repository root before retrying:

```powershell
cd C:\sacredloop\kv-cache-benchmark
```

**Permission denied on `Add-MpPreference`**
- PowerShell is not running as Administrator.
- Right-click PowerShell → Run as Administrator, then re-run Block 2.
- If admin access is unavailable, skip Block 2 and use the WSL2 fallback below.

**Permission denied on `git clone`**
- You are running from a protected directory.
- Block 1 creates `C:\sacredloop` first — ensure you ran that step before cloning.

**`Activate.ps1` cannot be loaded, running scripts is disabled**
- Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Then re-run `.venv\Scripts\Activate.ps1`

**DLL load failed: Application Control policy has blocked this file**
- Windows Defender blocked a compiled extension during install.
- Nuke the venv, ensure Block 2 ran as Administrator, then restart from Block 3.
- If the problem persists, use the WSL2 fallback.

**Hard gate prints `STOP: CUDA not available`**

```powershell
python -c "import torch; print(torch.__version__)"
```

If it shows `+cpu` instead of `+cu128`:

```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Re-run the hard gate. If it still fails, nuke the venv and restart from Block 3, ensuring Block 5 runs before Block 6.

**CUDA index out of bounds / model exceeded maximum length (gpt2 only)**
- GPT-2's context window is hard-capped at 1024 tokens total (prompt + generated).
- Do not increase `prompt_lengths` or `max_new_tokens` beyond the values in Block 8.
- The gpt2 run uses `64,128,256,512` prompt lengths and `max_new_tokens 256` specifically to stay within this limit.

**Out of memory on gpt-j-6B**
- gpt-j-6B requires approximately 12–16 GB VRAM in fp16.
- Check available VRAM:
  ```powershell
  python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1024**3)"
  ```
- If VRAM is insufficient, skip the gpt-j-6B run — the 3-model curve (gpt2, 1.3B, 2.7B) still demonstrates the finding.

**`FileNotFoundError: scripts/kv_receipt_basic_v2.py`**
- You are not in the repository root. Run `cd C:\sacredloop\kv-cache-benchmark` and retry.

**WSL2 fallback (Windows Defender issues or no admin access)**
- Open PowerShell and run: `wsl`
- Follow the Linux setup steps inside WSL2.
- Your NVIDIA GPU is accessible from WSL2 via driver passthrough.

