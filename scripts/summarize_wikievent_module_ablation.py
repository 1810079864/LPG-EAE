#!/usr/bin/env python3
import argparse
import re
import statistics
from pathlib import Path


DEV_RE = re.compile(r"current best dev-f1 score:\s*([0-9.eE+-]+)")
TEST_RE = re.compile(r"current related test-f1 score:\s*([0-9.eE+-]+)")


def read_result(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    dev = DEV_RE.findall(text)
    test = TEST_RE.findall(text)
    if not dev or not test:
        return None
    return float(dev[-1]), float(test[-1])


def mean_std(values):
    return statistics.mean(values), statistics.pstdev(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        default="exps_module_ablation/wikievent/full_model",
    )
    parser.add_argument(
        "--ablation-root", default="exps_module_ablation/wikievent"
    )
    parser.add_argument("--lr", default="2e-5")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[22, 42, 66, 99]
    )
    args = parser.parse_args()

    groups = [
        ("full_model", Path(args.baseline_root)),
        ("without_light_prompt", Path(args.ablation_root) / "without_light_prompt"),
        ("without_trigger_graph", Path(args.ablation_root) / "without_trigger_graph"),
        (
            "without_dependency_coreference_graph",
            Path(args.ablation_root) / "without_dependency_coreference_graph",
        ),
        ("without_entire_graph", Path(args.ablation_root) / "without_entire_graph"),
    ]

    results = {}
    print(f"{'group':42s} {'seed':>6s} {'dev_f1':>10s} {'test_f1':>10s}")
    print("-" * 74)
    for group, root in groups:
        rows = []
        for seed in args.seeds:
            path = root / str(seed) / args.lr / "log.txt"
            result = read_result(path) if path.exists() else None
            if result is None:
                print(f"{group:42s} {seed:6d} {'MISSING':>21s}")
                continue
            dev, test = result
            rows.append((seed, dev, test))
            print(f"{group:42s} {seed:6d} {dev:10.6f} {test:10.6f}")
        results[group] = rows

    print("\nFour-seed summary")
    print(f"{'group':42s} {'dev_f1 mean+/-std':>22s} {'test_f1 mean+/-std':>22s}")
    print("-" * 90)
    for group, _ in groups:
        rows = results[group]
        if len(rows) != len(args.seeds):
            print(f"{group:42s} {'INCOMPLETE':>22s} {'INCOMPLETE':>22s}")
            continue
        dev_mean, dev_std = mean_std([row[1] for row in rows])
        test_mean, test_std = mean_std([row[2] for row in rows])
        print(
            f"{group:42s} {dev_mean:.6f}+/-{dev_std:.6f}"
            f" {test_mean:.6f}+/-{test_std:.6f}"
        )

    baseline = results["full_model"]
    if len(baseline) == len(args.seeds):
        base_by_seed = {seed: (dev, test) for seed, dev, test in baseline}
        print("\nPaired contribution (full - ablated; positive means the module helps)")
        print(f"{'module':42s} {'dev_delta':>12s} {'test_delta':>12s}")
        print("-" * 68)
        for group, _ in groups[1:]:
            rows = results[group]
            if len(rows) != len(args.seeds):
                continue
            dev_delta = statistics.mean(
                base_by_seed[seed][0] - dev for seed, dev, _ in rows
            )
            test_delta = statistics.mean(
                base_by_seed[seed][1] - test for seed, _, test in rows
            )
            print(f"{group:42s} {dev_delta:+12.6f} {test_delta:+12.6f}")


if __name__ == "__main__":
    main()
