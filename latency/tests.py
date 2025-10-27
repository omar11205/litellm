import numpy as np
from scipy import stats
import json

# assume you already have latencies lists
with open("results_unified.json", "r") as f:
    data = json.load(f)

proxy = np.array(data["total_proxy_latencies"])
openai = np.array(data["total_openai_latencies"])

print("Proxy sample size:", len(proxy))
print("OpenAI sample size:", len(openai))

# 1) T‐test (check equal variances first)
# Option A: assume equal variances
t_stat, p_val = stats.ttest_ind(proxy, openai, equal_var=True)
print("t‐test (equal var) → t = %.3f, p = %.6f" % (t_stat, p_val))

# Option B: Welch’s t‐test (non‐equal variances)
t_stat2, p_val2 = stats.ttest_ind(proxy, openai, equal_var=False)
print("Welch’s t‐test → t = %.3f, p = %.6f" % (t_stat2, p_val2))

# 2) Kolmogorov‐Smirnov test (two‐sample)
ks_stat, ks_p = stats.ks_2samp(proxy, openai)
print("KS test → statistic = %.3f, p = %.6f" % (ks_stat, ks_p))

# Interpretation
alpha = 0.05
print("\nInterpretation:")
if p_val < alpha:
    print(" → Reject null hypothesis for equal means (equal‐var t‐test)")
else:
    print(" → Cannot reject null for equal‐var t‐test")

if p_val2 < alpha:
    print(" → Reject null hypothesis for equal means (Welch’s test)")
else:
    print(" → Cannot reject null for Welch’s test")

if ks_p < alpha:
    print(" → Reject null hypothesis that samples come from same distribution (KS test)")
else:
    print(" → Cannot reject null that distributions are same (KS test)")