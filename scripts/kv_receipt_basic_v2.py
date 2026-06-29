import argparse
import csv
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def run_benchmark(model_name, prompt_lengths, max_new_tokens, output_csv, device):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    base_text = (
        "The relationship between memory hierarchy design and inference efficiency "
        "is a foundational concern for hardware architects working on AI accelerators. "
    )
    rows = []
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for target_len in prompt_lengths:
        repeated = base_text * (target_len // 20 + 1)
        tokens = tokenizer(
            repeated, return_tensors="pt", truncation=True, max_length=target_len
        )
        input_ids = tokens["input_ids"].to(device)
        actual_prompt_tokens = input_ids.shape[1]
        print(f"\nmodel={model_name} device={device} prompt_tokens={actual_prompt_tokens}")

        print("--- RECOMPUTE PATH ---")
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=False,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        recompute_time = time.perf_counter() - t0
        print(f"time_s={recompute_time:.3f}")

        print("--- REUSE PATH ---")
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        reuse_time = time.perf_counter() - t1
        print(f"time_s={reuse_time:.3f}")

        speedup = recompute_time / reuse_time
        wasted = recompute_time - reuse_time
        print("--- DELTA ---")
        print(f"speedup_x={speedup:.3f}")
        print(f"wasted_s={wasted:.3f}")
        print(f"Receipt written -> {output_csv}")

        rows.append({
            "model": model_name,
            "device": device,
            "prompt_tokens": actual_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "recompute_time_s": round(recompute_time, 3),
            "reuse_time_s": round(reuse_time, 3),
            "speedup_x": round(speedup, 3),
            "wasted_compute_s": round(wasted, 3),
        })

    fieldnames = [
        "model", "device", "prompt_tokens", "max_new_tokens",
        "recompute_time_s", "reuse_time_s", "speedup_x", "wasted_compute_s",
    ]
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nAll results written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache Recompute vs Reuse Benchmark")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument(
        "--prompt_lengths",
        default="128,256,512,1024",
        help="Comma-separated list of target prompt token lengths",
    )
    parser.add_argument("--max_new_tokens", type=int, default=900)
    parser.add_argument("--output_csv", default="telemetry/kv_receipts.csv")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    lengths = [int(x) for x in args.prompt_lengths.split(",")]
    run_benchmark(args.model, lengths, args.max_new_tokens, args.output_csv, args.device)
