import json

timings = []
with open("timinglogLimit.txt", 'r') as f:
    timings = json.loads(f.read().replace("\'", "\""))

diffs = [1000 * (t1["timingDiff"] - t0["timingDiff"]) for t1, t0 in zip(timings[1:], timings[:-1])]

print("Avg diff:", sum(diffs) / len(diffs))
