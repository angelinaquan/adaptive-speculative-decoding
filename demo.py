"""Live verification-width example and an independent replay of the serving results."""
import argparse
import json
import sys
from pathlib import Path
from statistics import median
from zipfile import ZipFile

BATCH = 64

def probability(value):
    p = float(value)
    if not 0 <= p <= 1:
        raise ValueError("Use a probability between 0 and 1.")
    return p

def choose_width(costs, p):
    samples = costs["raw_target_samples"]
    sizes = sorted({size for size, _ in samples})
    medians = {size: median(t for n, t in samples if n == size) for size in sizes}
    draft_ms = costs["cost_tables"][0][BATCH]
    scores = []
    print(f"\nConditional acceptance p = {p:.2f}   |   {BATCH} requests, one common width")
    print(" Width   Expected tokens/request    Native target ms    Measured target ms")
    for width in range(1, 8):
        size = next(n for n in sizes if n >= BATCH * (1 + width))
        measured = medians[size]
        native = max(medians[n] for n in sizes if n <= size)
        reward = sum(p**i for i in range(width + 1))
        scores.append((reward/(draft_ms+native), reward/(draft_ms+measured)))
        print(f"{width:6} {reward:25.3f} {native:19.2f} {measured:21.2f}")
    winners = [1 + max(range(7), key=lambda i: scores[i][col]) for col in (0, 1)]
    print(f"Best expected tokens/ms: running maximum = width {winners[0]}; per-size median = width {winners[1]}.")
    return winners

def replay(archive):
    def read(name):
        return json.loads(archive.read(name))
    print("\nRecorded H200 experiment: paired outputs, four engine rounds")
    print(" Batch    Native tok/s    Measured tok/s    Throughput gain")
    for batch, version, ref in [(32, "v26", "ar-round0"), (64, "v38", "ar-reference")]:
        reference = read(f"cost_{version}_{ref}.json")["outputs"]
        rates = []
        for role in ("original", "measured"):
            tokens = seconds = 0
            for rnd in range(4):
                run = read(f"cost_{version}_{role}-round{rnd}.json")
                if run["status"] != "completed" or run["outputs"] != reference:
                    raise ValueError(f"Output check failed for {role}, round {rnd}")
                tokens += sum(row["tokens"] for row in run["records"])
                seconds += sum(row["seconds"] for row in run["records"])
            rates.append(tokens / seconds)
        print(f"{batch:6} {rates[0]:15,.2f} {rates[1]:17,.2f} {100*(rates[1]/rates[0]-1):+17.2f}%")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=probability, nargs="+", help="Run these probabilities without prompting.")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    archive_path = folder/"results"/"sessions.zip"
    if not archive_path.exists():
        archive_path = folder/"sessions.zip"
    with ZipFile(archive_path) as archive:
        costs = json.loads(archive.read("cost_v38_measured-round0.json"))["costs"]
        print("Live budget calculation using a real batch-64 H200 profile.")
        print("Score = expected committed tokens / (draft time + estimated verification time).")
        print("Simplifying assumption: one conditional acceptance probability and one width for all requests.")
        if args.p or not sys.stdin.isatty():
            for p in args.p or [0.95, 0.40]:
                choose_width(costs, p)
        else:
            choose_width(costs, 0.95)
            while True:
                try:
                    entry = input("\nTry a new p [Enter for 0.40], or type results: ").strip()
                except EOFError:
                    break
                if entry.lower() in {"results", "r", "q", "quit"}:
                    break
                try:
                    choose_width(costs, probability(entry or "0.40"))
                except ValueError as err:
                    print(err)
        replay(archive)

if __name__ == "__main__":
    main()
