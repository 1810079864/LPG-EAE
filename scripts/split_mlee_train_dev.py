#!/usr/bin/env python3
"""Create a reproducible, document-disjoint MLEE train/dev split."""

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

SPLIT_VERSION = 2


def document_id(record):
    sample_id = str(record["id"])
    return sample_id.split("-", 1)[1] if "-" in sample_id else sample_id


def event_counts(records):
    counts = collections.Counter()
    for record in records:
        counts.update(event["event_type"] for event in record.get("events", []))
    return counts


def split_score(selected, groups, total_events, target_instances, ratio):
    records = [row for key in selected for row in groups[key]]
    counts = event_counts(records)
    size_error = abs(len(records) - target_instances) / max(target_instances, 1)
    event_error = sum(
        abs(counts[event_type] - count * ratio) / max(count * ratio, 1.0)
        for event_type, count in total_events.items()
    ) / max(len(total_events), 1)
    # Strongly discourage dropping event types that are frequent enough to
    # reasonably occur in the development set.
    missing = sum(
        1 for event_type, count in total_events.items()
        if count * ratio >= 1.0 and counts[event_type] == 0
    )
    missing_from_train = sum(
        1 for event_type, count in total_events.items()
        if counts[event_type] >= count
    )
    return (
        size_error + event_error
        + 2.0 * missing
        + 10.0 * missing_from_train
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    train_output = Path(args.train_output)
    dev_output = Path(args.dev_output)
    manifest_path = Path(args.manifest)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    if not args.force and train_output.exists() and dev_output.exists() \
            and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_sha256") == source_hash
            and manifest.get("split_version") == SPLIT_VERSION
            and manifest.get("seed") == args.seed
            and manifest.get("dev_ratio") == args.dev_ratio
            and manifest.get("trials") == args.trials
        ):
            print(
                "[MLEE split] Reusing document-disjoint split: "
                f"train={manifest['train_instances']}, "
                f"dev={manifest['dev_instances']}"
            )
            return

    with source.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]

    groups = collections.defaultdict(list)
    for record in records:
        groups[document_id(record)].append(record)
    document_ids = sorted(groups)
    dev_document_count = max(1, round(len(document_ids) * args.dev_ratio))
    target_instances = round(len(records) * args.dev_ratio)
    total_events = event_counts(records)

    rng = random.Random(args.seed)
    best_documents = None
    best_score = float("inf")
    for _ in range(args.trials):
        selected = tuple(rng.sample(document_ids, dev_document_count))
        score = split_score(
            selected, groups, total_events, target_instances, args.dev_ratio
        )
        if score < best_score:
            best_score = score
            best_documents = set(selected)

    train_records = [
        record for record in records
        if document_id(record) not in best_documents
    ]
    dev_records = [
        record for record in records
        if document_id(record) in best_documents
    ]
    train_documents = {document_id(record) for record in train_records}
    dev_documents = {document_id(record) for record in dev_records}
    if train_documents & dev_documents:
        raise RuntimeError("Document leakage detected in MLEE split.")

    for path, rows in ((train_output, train_records), (dev_output, dev_records)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "split_version": SPLIT_VERSION,
        "source": str(source),
        "source_sha256": source_hash,
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "trials": args.trials,
        "train_documents": len(train_documents),
        "dev_documents": len(dev_documents),
        "train_instances": len(train_records),
        "dev_instances": len(dev_records),
        "train_event_counts": dict(sorted(event_counts(train_records).items())),
        "dev_event_counts": dict(sorted(event_counts(dev_records).items())),
        "split_score": best_score,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[MLEE split] Created document-disjoint split: "
        f"train={len(train_records)} instances/{len(train_documents)} documents, "
        f"dev={len(dev_records)} instances/{len(dev_documents)} documents"
    )


if __name__ == "__main__":
    main()
