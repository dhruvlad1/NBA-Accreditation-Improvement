import pandas as pd
from scipy.stats import mannwhitneyu

d = pd.read_csv("results/ablation_runs.csv")
print("columns:", list(d.columns))
vcol = "variant"
cols = [c for c in d.columns if any(k in c for k in ("min_cost", "min_hours", "max_f4", "igd"))]
for col in cols:
    base = d[d[vcol] == "V0"][col].dropna()
    for v in ["V1", "V2", "V3"]:
        x = d[d[vcol] == v][col].dropna()
        print(f"{col:20s} {v}  mean={x.mean():.4f} vs V0 {base.mean():.4f}  p={mannwhitneyu(x, base).pvalue:.4f}")