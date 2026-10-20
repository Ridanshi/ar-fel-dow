import glob, json
import numpy as np
R = [json.load(open(f)) for f in sorted(glob.glob("results/results_pitfall_ablation_seed*.json"))]
print("seeds", [r["seed"] for r in R])
def ms(v):
    v = np.array(v, float) * 100
    return "%.2f +/- %.2f" % (v.mean(), v.std())
for k in ["row_with_count", "group_no_count", "row_no_count"]:
    for m in ["acc", "attack_recall", "benign_recall"]:
        print(k, m, ms([r[k][m] for r in R]))
for k in ["plain_pgd", "plain_gan", "reuse_pgd", "reuse_gan", "reuse_clean"]:
    print(k, ms([r[k] for r in R]))
