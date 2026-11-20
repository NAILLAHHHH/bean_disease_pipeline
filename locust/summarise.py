"""Turn Locust CSVs into a markdown table + chart (locust/results/)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "results"
rows = []
for n in (1, 2, 4):
    f = R / f"c{n}_stats.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f).set_index("Name")
    p, agg = d.loc["/predict"], d.loc["Aggregated"]
    rows.append({"containers": n, "requests": int(agg["Request Count"]), "failures": int(agg["Failure Count"]),
                 "req/s": round(agg["Requests/s"], 1), "avg ms": round(p["Average Response Time"]),
                 "median ms": round(p["Median Response Time"]), "p95 ms": round(p["95%"]),
                 "p99 ms": round(p["99%"]), "max ms": round(p["Max Response Time"])})
df = pd.DataFrame(rows)
md = df.to_markdown(index=False)
(R / "summary.md").write_text(md + "\n")
print(md)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
x = df["containers"].astype(str)
for col, c, ls in (("median ms", "#2a78d6", "-"), ("p95 ms", "#eb6834", "-"), ("avg ms", "#7a7974", "--")):
    axes[0].plot(x, df[col], ls, marker="o", lw=2, color=c, label=col.replace(" ms", ""))
axes[0].set(title="/predict response time", xlabel="API containers", ylabel="ms"); axes[0].legend()
axes[1].bar(x, df["req/s"], color="#2a78d6", width=.5)
for i, v in enumerate(df["req/s"]):
    axes[1].text(i, v, f"{v}", ha="center", va="bottom")
axes[1].set(title="Throughput", xlabel="API containers", ylabel="requests / second")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(R / "load_test.png", dpi=130)
