"""
Quick rolling-average loss viewer for train_log.jsonl (written by
step2_finetune_refined.py / step3_finetune_hallucination.py every optimizer
step). Per-step loss is noisy on short-answer VQA data -- this smooths it out
so you can see the actual trend without relying on wandb's UI.

Run:
    python analyze_train_log.py checkpoints/vlm_kvasir_full/train_log.jsonl
    python analyze_train_log.py checkpoints/vlm_kvasir_full/train_log.jsonl --window 100
"""

import argparse
import json
from collections import deque


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_path")
    p.add_argument("--window", type=int, default=50, help="Rolling average window (steps)")
    p.add_argument("--print_every", type=int, default=200, help="Print a rolling-avg line every N steps")
    args = p.parse_args()

    window = deque(maxlen=args.window)
    epoch_sums = {}
    epoch_counts = {}
    epoch_val_loss = {}
    step_evals = []   # mid-epoch val readings: (step, epoch, val_loss, is_best)

    with open(args.log_path) as f:
        for line in f:
            entry = json.loads(line)

            if "epoch_summary" in entry:
                epoch_val_loss[entry["epoch_summary"]] = entry.get("val_loss")
                continue

            if "step_eval" in entry:
                step_evals.append((entry["step_eval"], entry["epoch"],
                                   entry["val_loss"], entry.get("is_best", False)))
                continue

            step = entry["step"]
            loss = entry["loss"]
            epoch = entry["epoch"]

            window.append(loss)
            epoch_sums[epoch] = epoch_sums.get(epoch, 0.0) + loss
            epoch_counts[epoch] = epoch_counts.get(epoch, 0) + 1

            if step % args.print_every == 0:
                avg = sum(window) / len(window)
                print(f"step {step:>6} | epoch {epoch} | rolling_avg(last {len(window)}) = {avg:.4f} | last_lr = {entry['lr']:.2e}")

    print("\nPer-epoch average loss:")
    for epoch in sorted(epoch_sums):
        avg = epoch_sums[epoch] / epoch_counts[epoch]
        val = epoch_val_loss.get(epoch)
        val_str = f" | val_loss = {val:.4f}" if val is not None else ""
        print(f"  epoch {epoch}: train_avg = {avg:.4f}  ({epoch_counts[epoch]} steps){val_str}")

    if step_evals:
        print("\nMid-epoch val_loss readings:")
        for s, ep, vl, best in sorted(step_evals):
            best_str = " <- BEST" if best else ""
            print(f"  step {s:>6} | epoch {ep} | val_loss = {vl:.4f}{best_str}")


if __name__ == "__main__":
    main()
