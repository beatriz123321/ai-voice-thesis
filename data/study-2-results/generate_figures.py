import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy import stats
import re, warnings
warnings.filterwarnings('ignore')

# ── palette ───────────────────────────────────────────────────────────────────
N_COLOR  = "#4C72B0"   # Condition N → blue
E_COLOR  = "#DD8452"   # Condition E → orange
NEUTRAL  = "#8C8C8C"
BG       = "#FAFAFA"
GRID     = "#E4E4E4"
DARK     = "#222222"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#CCCCCC", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

OUT  = "/Users/beatrizlopes/Downloads/thesis_v2/"
FILE = "/Users/beatrizlopes/Downloads/full_responses_wide.xlsx"

# ── name mappings (data uses "Version A/B"; display uses "Condition N/E") ─────
VER_LABEL = {"Version A": "Condition N", "Version B": "Condition E"}
VER_COLOR = {"Version A": N_COLOR, "Version B": E_COLOR}
COND_N, COND_E = "Condition N", "Condition E"

LIKERT = {
    "Strongly Disagree":1,"Disagree":2,"Somewhat Disagree":3,
    "Neither Agree nor Disagree":4,"Somewhat Agree":5,"Agree":6,"Strongly Agree":7
}
TRAITS = ["Warm","Kind","Friendly","Relaxing","Calming","Knowledgeable",
          "Competent","Accurate","Trustworthy","Reliable","Dependable",
          "Intelligent","Practical","Human-like","Machine-like","Natural","Fake"]

df = pd.read_excel(FILE)
for t in TRAITS:
    df[f"S2S3_{t}_n"] = df[f"S2S3_{t}"].map(LIKERT)
    df[f"S4_{t}_n"]   = df[f"S4_{t}"].map(LIKERT)

def map_pref(row):
    if "first"  in str(row["preferred_version"]).lower(): return VER_LABEL[row["S2S3_version"]]
    if "second" in str(row["preferred_version"]).lower(): return VER_LABEL[row["S4_version"]]
    return "No preference"
df["pref_actual"] = df.apply(map_pref, axis=1)

# long form – using display labels
rows = []
for _, r in df.iterrows():
    for t in TRAITS:
        rows.append({"condition": VER_LABEL[r["S2S3_version"]], "survey":"First",  "trait":t, "score":r[f"S2S3_{t}_n"]})
        rows.append({"condition": VER_LABEL[r["S4_version"]],   "survey":"Second", "trait":t, "score":r[f"S4_{t}_n"]})
long = pd.DataFrame(rows)

def paired_ttest(df, trait):
    """Paired t-test: Condition N (Version A) vs Condition E (Version B)."""
    n_scores, e_scores = [], []
    for _, row in df.iterrows():
        if row["S2S3_version"] == "Version A":
            n_scores.append(row[f"S2S3_{trait}_n"])
            e_scores.append(row[f"S4_{trait}_n"])
        else:
            e_scores.append(row[f"S2S3_{trait}_n"])
            n_scores.append(row[f"S4_{trait}_n"])
    pairs = pd.DataFrame({"n": n_scores, "e": e_scores}).dropna()
    return stats.ttest_rel(pairs["n"], pairs["e"])

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

ver_means = long.groupby(["condition","trait"])["score"].mean().unstack("condition")
ver_sds   = long.groupby(["condition","trait"])["score"].std().unstack("condition")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 – Demographics
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Participant Demographics  (N = 48)", fontsize=15, fontweight="bold", y=1.02)

ax = axes[0]
ages = df["age"].dropna()
ax.hist(ages, bins=9, color=N_COLOR, edgecolor="white", linewidth=0.9)
ax.axvline(ages.mean(),   color="#C44E52", lw=2, ls="--", label=f"Mean = {ages.mean():.1f}")
ax.axvline(ages.median(), color="#55A868", lw=2, ls=":",  label=f"Median = {ages.median():.0f}")
ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.set_title("Age Distribution"); ax.legend()

ax = axes[1]
gc = df["gender"].value_counts()
wedges, texts, autos = ax.pie(gc.values, labels=gc.index, autopct="%1.0f%%",
    colors=[N_COLOR, E_COLOR, NEUTRAL][:len(gc)], startangle=90,
    wedgeprops={"edgecolor":"white","linewidth":2})
for at in autos: at.set_fontsize(11)
ax.set_title("Gender Distribution")

ax = axes[2]
eng = df["english_level"].value_counts()
short = {"Fluent / Proficient (C1 or C2)":"Fluent (C1/C2)",
         "Intermediate / Independent (B1 or B2)":"Intermediate (B1/B2)",
         "Native speaker":"Native"}
bars = ax.barh([short.get(l,l) for l in eng.index], eng.values,
               color=[N_COLOR,E_COLOR,NEUTRAL][:len(eng)], edgecolor="white")
for bar, v in zip(bars, eng.values):
    ax.text(v+0.3, bar.get_y()+bar.get_height()/2, str(v), va="center", fontweight="bold")
ax.set_xlabel("Count"); ax.set_xlim(0, max(eng.values)+6); ax.set_title("English Proficiency")

plt.tight_layout()
plt.savefig(OUT+"fig01_demographics.png"); plt.close()
print("✓ fig01_demographics.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 – Preference
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Participant Condition Preference", fontsize=15, fontweight="bold", y=1.02)

ax = axes[0]
pref = df["pref_actual"].value_counts()
cmap = {COND_N: N_COLOR, COND_E: E_COLOR, "No preference": NEUTRAL}
wedges, texts, autos = ax.pie(pref.values, labels=pref.index, autopct="%1.0f%%",
    colors=[cmap[k] for k in pref.index], startangle=90,
    wedgeprops={"edgecolor":"white","linewidth":2})
for at in autos: at.set_fontsize(12); at.set_fontweight("bold")
ax.set_title(f"Overall Preference  (N = {len(df)})")

ax = axes[1]
n_first = len(df[df["S2S3_version"]=="Version A"])
e_first = len(df[df["S2S3_version"]=="Version B"])
xlabels = [f"Heard {COND_N} First\n(n={n_first})", f"Heard {COND_E} First\n(n={e_first})"]
groups  = [df[df["S2S3_version"]=="Version A"], df[df["S2S3_version"]=="Version B"]]
x = np.arange(2); w = 0.55; bottoms = np.zeros(2)
for label, col in [(COND_N, N_COLOR), (COND_E, E_COLOR), ("No preference", NEUTRAL)]:
    vals = np.array([(g["pref_actual"]==label).sum() for g in groups], dtype=float)
    bars = ax.bar(x, vals, w, bottom=bottoms, color=col, edgecolor="white", label=label)
    for bar, v, bot in zip(bars, vals, bottoms):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bot+v/2, str(int(v)),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    bottoms += vals
ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=10)
ax.set_ylabel("Number of Participants"); ax.set_title("Preference by Presentation Order")
ax.legend(loc="upper right"); ax.set_ylim(0, max(bottoms)+4)

plt.tight_layout()
plt.savefig(OUT+"fig02_preference.png"); plt.close()
print("✓ fig02_preference.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 – Noticed difference
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
order = ["Not at all","Little","Somewhat","Very likely","To a great extent"]
counts = df["noticed_difference"].value_counts().reindex(order, fill_value=0)
palette = plt.cm.RdYlGn(np.linspace(0.1, 0.85, len(order)))
bars = ax.bar(order, counts.values, color=palette, edgecolor="white", width=0.6)
total = len(df)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{v}\n({v/total*100:.0f}%)", ha="center", fontweight="bold", fontsize=10)
ax.set_ylabel("Number of Participants"); ax.set_ylim(0, max(counts.values)+6)
ax.set_title("To What Extent Did Participants Notice a Difference\nBetween the Two Conditions?")
plt.tight_layout()
plt.savefig(OUT+"fig03_noticed_difference.png"); plt.close()
print("✓ fig03_noticed_difference.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 – Grouped bar: all 17 traits
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(17, 6.5))
x = np.arange(len(TRAITS)); w = 0.36
bars_n = ax.bar(x-w/2, [ver_means[COND_N][t] for t in TRAITS], w,
                yerr=[ver_sds[COND_N][t] for t in TRAITS], capsize=3,
                color=N_COLOR, edgecolor="white", label=COND_N,
                error_kw={"elinewidth":1,"ecolor":"#555555"})
bars_e = ax.bar(x+w/2, [ver_means[COND_E][t] for t in TRAITS], w,
                yerr=[ver_sds[COND_E][t] for t in TRAITS], capsize=3,
                color=E_COLOR, edgecolor="white", label=COND_E,
                error_kw={"elinewidth":1,"ecolor":"#555555"})

for i, t in enumerate(TRAITS):
    tstat, p = paired_ttest(df, t)
    sl = sig_label(p)
    if sl != "ns":
        y_max = max(ver_means[COND_N][t]+ver_sds[COND_N][t],
                    ver_means[COND_E][t]+ver_sds[COND_E][t]) + 0.3
        ax.plot([i-w/2, i+w/2], [y_max, y_max], color=DARK, lw=0.9)
        ax.text(i, y_max+0.08, sl, ha="center", fontsize=9, color=DARK)

ax.axhline(4, color=NEUTRAL, lw=1, ls="--", alpha=0.7, label="Neutral midpoint (4)")
ax.set_xticks(x); ax.set_xticklabels(TRAITS, rotation=38, ha="right")
ax.set_ylabel("Mean Score (1–7 Likert)"); ax.set_ylim(1, 8.5)
ax.set_title(f"Mean Trait Ratings: {COND_N} vs {COND_E}\n(Error bars = SD;  * p<.05  ** p<.01  *** p<.001, paired t-test, df=47)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT+"fig04_all_traits.png"); plt.close()
print("✓ fig04_all_traits.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 – Radar chart
# ─────────────────────────────────────────────────────────────────────────────
RADAR = ["Warm","Kind","Friendly","Relaxing","Calming","Knowledgeable",
         "Competent","Accurate","Trustworthy","Reliable","Dependable",
         "Intelligent","Practical","Human-like","Natural"]
N_r = len(RADAR)
angles = np.linspace(0, 2*np.pi, N_r, endpoint=False).tolist() + [0]
mN = [ver_means[COND_N][t] for t in RADAR] + [ver_means[COND_N][RADAR[0]]]
mE = [ver_means[COND_E][t] for t in RADAR] + [ver_means[COND_E][RADAR[0]]]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection":"polar"})
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(angles, mN, color=N_COLOR, lw=2.2); ax.fill(angles, mN, color=N_COLOR, alpha=0.18)
ax.plot(angles, mE, color=E_COLOR, lw=2.2); ax.fill(angles, mE, color=E_COLOR, alpha=0.18)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(RADAR, size=10)
ax.set_ylim(1,7); ax.set_yticks([2,3,4,5,6,7])
ax.set_yticklabels(["2","3","4","5","6","7"], size=8, color="grey")
ax.grid(color=GRID, lw=0.8)
ax.legend(handles=[mpatches.Patch(color=N_COLOR,alpha=0.6,label=COND_N),
                   mpatches.Patch(color=E_COLOR,alpha=0.6,label=COND_E)],
          loc="upper right", bbox_to_anchor=(1.32,1.12), fontsize=11)
ax.set_title(f"Trait Profile Comparison: {COND_N} vs {COND_E}\n(Mean ratings, 1–7 scale)",
             size=13, fontweight="bold", pad=22)
plt.tight_layout()
plt.savefig(OUT+"fig05_radar.png"); plt.close()
print("✓ fig05_radar.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 – Perceived Warmth / Perceived Trust / Perceived Realism
# ─────────────────────────────────────────────────────────────────────────────
dims = {
    "Perceived Comfort":          ["Warm","Kind","Friendly","Relaxing","Calming"],
    "Perceived Trustworthiness":  ["Knowledgeable","Competent","Accurate","Trustworthy","Reliable","Dependable","Intelligent","Practical"],
    "Perceived Anthropomorphism": ["Human-like","Natural"],
}
dim_means_v, dim_sds_v, dim_ps = {}, {}, {}
for dim, tlist in dims.items():
    for cond, ver in [(COND_N,"Version A"), (COND_E,"Version B")]:
        sc = []
        for _, row in df.iterrows():
            for sfx in ["S2S3","S4"]:
                vc = "S2S3_version" if sfx=="S2S3" else "S4_version"
                if row[vc]==ver:
                    sc += [row[f"{sfx}_{t}_n"] for t in tlist]
        s = pd.Series(sc).dropna()
        dim_means_v[(dim,cond)] = s.mean(); dim_sds_v[(dim,cond)] = s.std()
    a_all, b_all = [], []
    for _, row in df.iterrows():
        for sfx in ["S2S3","S4"]:
            vc = "S2S3_version" if sfx=="S2S3" else "S4_version"
            for t in tlist:
                if row[vc]=="Version A": a_all.append(row[f"{sfx}_{t}_n"])
                else:                    b_all.append(row[f"{sfx}_{t}_n"])
    _, p = stats.ttest_ind(pd.Series(a_all).dropna(), pd.Series(b_all).dropna())
    dim_ps[dim] = p

dnames = list(dims.keys())
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(dnames)); w = 0.36
for idx, (cond, col) in enumerate([(COND_N, N_COLOR), (COND_E, E_COLOR)]):
    vals = [dim_means_v[(d,cond)] for d in dnames]
    errs = [dim_sds_v[(d,cond)]  for d in dnames]
    bars = ax.bar(x+idx*w-w/2, vals, w, yerr=errs, capsize=4,
                  color=col, edgecolor="white", label=cond,
                  error_kw={"elinewidth":1.2,"ecolor":"#555555"})
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

for i, d in enumerate(dnames):
    p = dim_ps[d]; sl = sig_label(p)
    if sl != "ns":
        y_max = max(dim_means_v[(d,COND_N)]+dim_sds_v[(d,COND_N)],
                    dim_means_v[(d,COND_E)]+dim_sds_v[(d,COND_E)]) + 0.35
        ax.plot([i-w/2, i+w/2], [y_max,y_max], color=DARK, lw=1)
        ax.text(i, y_max+0.06, sl, ha="center", fontsize=11, color=DARK)

ax.axhline(4, color=NEUTRAL, lw=1, ls="--", alpha=0.7, label="Neutral (4)")
ax.set_xticks(x); ax.set_xticklabels(dnames, fontsize=12)
ax.set_ylabel("Mean Composite Score (1–7)"); ax.set_ylim(1, 7.2)
ax.set_title(f"Perceived Comfort, Perceived Trustworthiness & Perceived Anthropomorphism\n{COND_N} vs {COND_E}  (* p<.05  *** p<.001)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT+"fig06_dimensions.png"); plt.close()
print("✓ fig06_dimensions.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 – Impression order effect
# ─────────────────────────────────────────────────────────────────────────────
order_data = {}
for cond, ver in [(COND_N,"Version A"), (COND_E,"Version B")]:
    first_sc, second_sc = [], []
    for _, row in df.iterrows():
        for t in TRAITS:
            if row["S2S3_version"]==ver: first_sc.append(row[f"S2S3_{t}_n"])
            if row["S4_version"]==ver:   second_sc.append(row[f"S4_{t}_n"])
    f_s = pd.Series(first_sc).dropna(); s_s = pd.Series(second_sc).dropna()
    order_data[cond] = {"first_m":f_s.mean(),"first_sd":f_s.std(),
                        "second_m":s_s.mean(),"second_sd":s_s.std()}

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(2); w = 0.36
for idx, (cond, col) in enumerate([(COND_N, N_COLOR), (COND_E, E_COLOR)]):
    d = order_data[cond]
    vals = [d["first_m"], d["second_m"]]
    errs = [d["first_sd"], d["second_sd"]]
    bars = ax.bar(x+idx*w-w/2, vals, w, yerr=errs, capsize=4,
                  color=col, edgecolor="white", label=cond,
                  error_kw={"elinewidth":1.2,"ecolor":"#555555"})
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

n_f = pd.Series([row[f"S2S3_{t}_n"] for _, row in df.iterrows() if row["S2S3_version"]=="Version A" for t in TRAITS]).dropna()
n_s = pd.Series([row[f"S4_{t}_n"]   for _, row in df.iterrows() if row["S4_version"]=="Version A"   for t in TRAITS]).dropna()
_, p_n = stats.ttest_ind(n_f, n_s)
y_top = order_data[COND_N]["first_m"] + order_data[COND_N]["first_sd"] + 0.5
ax.plot([-w/2, w/2], [y_top, y_top], color=N_COLOR, lw=1.2)
ax.text(0, y_top+0.06, sig_label(p_n), ha="center", fontsize=11, color=N_COLOR)

ax.axhline(4, color=NEUTRAL, lw=1, ls="--", alpha=0.7, label="Neutral (4)")
ax.set_xticks(x); ax.set_xticklabels(["First Impression\n(S2/S3 survey)", "Second Impression\n(S4 survey)"])
ax.set_ylabel("Mean Score (1–7 Likert)"); ax.set_ylim(1, 6.5)
ax.set_title("Impression Order Effect\nDoes Hearing a Condition Second Change Its Rating?")
ax.legend()
plt.tight_layout()
plt.savefig(OUT+"fig07_order_effect.png"); plt.close()
print("✓ fig07_order_effect.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 – Heatmap
# ─────────────────────────────────────────────────────────────────────────────
heat = ver_means[[COND_N, COND_E]].T
fig, ax = plt.subplots(figsize=(18, 3.5))
sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn", vmin=1, vmax=7,
            linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label":"Mean Score (1–7)"})
ax.set_title(f"Heatmap of Mean Trait Ratings  —  {COND_N} vs {COND_E}  (All Rounds Combined)",
             fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUT+"fig08_heatmap.png"); plt.close()
print("✓ fig08_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 – Difference plot (E − N)
# ─────────────────────────────────────────────────────────────────────────────
res = []
for t in TRAITS:
    tstat, p = paired_ttest(df, t)
    diff = ver_means[COND_E][t] - ver_means[COND_N][t]
    res.append({"trait":t,"diff":diff,"p":p,"sig":sig_label(p)})
res_df = pd.DataFrame(res).sort_values("diff")

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = [E_COLOR if d>0 else N_COLOR for d in res_df["diff"]]
ax.barh(res_df["trait"], res_df["diff"], color=colors_bar, edgecolor="white", height=0.65)
ax.axvline(0, color=DARK, lw=1)
for i, (_, row) in enumerate(res_df.iterrows()):
    if row["sig"] != "ns":
        x_pos = row["diff"] + (0.06 if row["diff"]>=0 else -0.06)
        ha = "left" if row["diff"]>=0 else "right"
        ax.text(x_pos, i, row["sig"], va="center", ha=ha, fontsize=12, fontweight="bold")
ax.set_xlabel(f"Mean Difference  ({COND_E} − {COND_N})", fontsize=11)
ax.set_title(f"Trait Rating Differences: {COND_E} minus {COND_N}\n(* p<.05  ** p<.01  *** p<.001, paired t-test, N=48, df=47)")
ax.legend(handles=[mpatches.Patch(color=E_COLOR, label=f"{COND_E} rated higher"),
                   mpatches.Patch(color=N_COLOR, label=f"{COND_N} rated higher")], loc="lower right")
plt.tight_layout()
plt.savefig(OUT+"fig09_difference_plot.png"); plt.close()
print("✓ fig09_difference_plot.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 – Likert distribution strips for 10 key traits
# ─────────────────────────────────────────────────────────────────────────────
KEY = ["Friendly","Warm","Kind","Trustworthy","Reliable","Natural","Human-like","Machine-like","Knowledgeable","Practical"]
LCOLORS = ["#d7191c","#f17c4a","#fec981","#ffffbf","#a6d96a","#4dac26","#1a7837"]
LLABELS = ["Strongly\nDisagree","Disagree","Somewhat\nDisagree","Neutral","Somewhat\nAgree","Agree","Strongly\nAgree"]

fig, axes = plt.subplots(len(KEY), 2, figsize=(13, len(KEY)*1.5), sharex=True)
fig.suptitle(f"Likert Response Distributions for Key Traits  ({COND_N} vs {COND_E})",
             fontsize=13, fontweight="bold", y=1.005)

for ri, t in enumerate(KEY):
    for ci, (cond, col) in enumerate([(COND_N, N_COLOR), (COND_E, E_COLOR)]):
        ax = axes[ri, ci]
        data = long[(long["condition"]==cond)&(long["trait"]==t)]["score"].dropna()
        cnts = [(data==s).sum() for s in range(1,8)]
        total = sum(cnts); left = 0
        for s, c, lc in zip(range(1,8), cnts, LCOLORS):
            if c > 0:
                ax.barh(0, c/total, left=left, color=lc, height=0.7, edgecolor="white")
                if c/total > 0.07:
                    ax.text(left+c/total/2, 0, f"{c/total*100:.0f}%",
                            ha="center", va="center", fontsize=8)
                left += c/total
        ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xticks([]); ax.grid(False)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        ax.axvline(0.5, color="white", lw=0.8, ls="--", alpha=0.5)
        if ri == 0: ax.set_title(cond, color=col, fontweight="bold", fontsize=11)
        ax.set_ylabel(t, rotation=0, labelpad=65, va="center", fontsize=9.5)

fig.legend(handles=[mpatches.Patch(color=c,label=l) for c,l in zip(LCOLORS,LLABELS)],
           loc="lower center", ncol=7, bbox_to_anchor=(0.5,-0.02), fontsize=8.5)
plt.tight_layout()
plt.savefig(OUT+"fig10_likert_strips.png"); plt.close()
print("✓ fig10_likert_strips.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 – Stats table
# ─────────────────────────────────────────────────────────────────────────────
tbl_rows = []
for t in TRAITS:
    n_sc = long[(long["condition"]==COND_N)&(long["trait"]==t)]["score"].dropna()
    e_sc = long[(long["condition"]==COND_E)&(long["trait"]==t)]["score"].dropna()
    tstat, p = paired_ttest(df, t)
    diff = e_sc.mean()-n_sc.mean()
    tbl_rows.append({"Trait": t,
                     f"{COND_N}  M (SD)": f"{n_sc.mean():.2f} ({n_sc.std():.2f})",
                     f"{COND_E}  M (SD)": f"{e_sc.mean():.2f} ({e_sc.std():.2f})",
                     f"{COND_E} − {COND_N}": f"{diff:+.2f}",
                     "t(47)": f"{tstat:.2f}",
                     "p": f"{p:.3f}",
                     "": sig_label(p)})
tbl = pd.DataFrame(tbl_rows)

fig, ax = plt.subplots(figsize=(14, 6.5))
ax.axis("off")
table = ax.table(cellText=tbl.values, colLabels=tbl.columns, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1, 1.6)
for j in range(len(tbl.columns)):
    table[0,j].set_facecolor("#2C3E50"); table[0,j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(tbl)+1):
    sl = tbl.iloc[i-1][""]
    bg = "#FFF3CD" if sl not in ("ns","") else ("#F5F5F5" if i%2==0 else "white")
    for j in range(len(tbl.columns)):
        table[i,j].set_facecolor(bg)
ax.set_title(f"Table 1. Descriptive Statistics and Paired t-Tests: {COND_N} vs {COND_E}  (N = 48, df = 47)",
             fontweight="bold", fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(OUT+"fig11_stats_table.png"); plt.close()
print("✓ fig11_stats_table.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 – Qualitative themes
# ─────────────────────────────────────────────────────────────────────────────
df["all_comments"] = (df["difference_description"].fillna("")+" "+df["preference_reason"].fillna("")).str.strip()
THEMES = {
    "Tone & Intonation":           r"tone|intonation|pitch|rhythm|rythm|pace|speed|slow|fast|monoton|volume|accent|pronounc",
    "Warmth & Friendliness":       r"warm|friend|kind|welcom|lively|energetic|upbeat|enthusias|excit|happy|relat|invit",
    "Naturalness":                 r"natural|human.?like|human\b|genuine|real|authentic|person|ease|easy to listen|approachab",
    "Roboticness / Artificiality": r"robot|machine|fake|artificial|programm|script|synthetic|forced|hollow|empty|word by word|mechanic",
    "Engagement & Motivation":     r"motivat|bored|engag|capti|fun|enjoyab|pleasant|effort|conscious|spontan|annoy",
    "Comfort & Calm":              r"calm|comfort|relax|sooth|pleasant|eas(y|ier)|smooth",
}
def code(text):
    text = str(text).lower()
    return [th for th, pat in THEMES.items() if re.search(pat, text)]

df["themes"] = df["all_comments"].apply(code)
theme_counts  = {t:0 for t in THEMES}
theme_by_pref = {t:{COND_N:0, COND_E:0} for t in THEMES}
for _, row in df.iterrows():
    for t in row["themes"]:
        theme_counts[t] += 1
        if row["pref_actual"] in (COND_N, COND_E):
            theme_by_pref[t][row["pref_actual"]] += 1

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.suptitle("Qualitative Themes in Open-Ended Participant Responses", fontsize=14, fontweight="bold", y=1.02)

ax = axes[0]
tc = pd.Series(theme_counts).sort_values(ascending=True)
bars = ax.barh(tc.index, tc.values, color=plt.cm.Set2(np.linspace(0,1,len(tc))), edgecolor="white", height=0.6)
for bar, v in zip(bars, tc.values):
    ax.text(v+0.2, bar.get_y()+bar.get_height()/2, str(v), va="center", fontweight="bold", fontsize=11)
ax.set_xlabel("Number of Participants"); ax.set_xlim(0, max(tc.values)+5)
ax.set_title("Theme Frequency")

ax = axes[1]
tp_df = pd.DataFrame(theme_by_pref).T.loc[tc.index]
x2 = np.arange(len(tp_df)); w2=0.36
bN = ax.barh(x2-w2/2, tp_df[COND_N], w2, color=N_COLOR, edgecolor="white", label=f"Preferred {COND_N}")
bE = ax.barh(x2+w2/2, tp_df[COND_E], w2, color=E_COLOR, edgecolor="white", label=f"Preferred {COND_E}")
for bar, v in zip(list(bN)+list(bE), list(tp_df[COND_N])+list(tp_df[COND_E])):
    if v>0: ax.text(v+0.1, bar.get_y()+bar.get_height()/2, str(v), va="center", fontsize=9, fontweight="bold")
ax.set_yticks(x2); ax.set_yticklabels(tp_df.index)
ax.set_xlabel("Number of Participants"); ax.legend(loc="lower right")
ax.set_title("Themes by Preferred Condition"); ax.set_xlim(0, max(tp_df.max())+5)

plt.tight_layout()
plt.savefig(OUT+"fig12_qualitative_themes.png"); plt.close()
print("✓ fig12_qualitative_themes.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 – Quote cards
# ─────────────────────────────────────────────────────────────────────────────
QUOTES = [
    ('"The first was warm and imagining the trip with\nyou, whereas the second didn\'t seem happy to\nbe doing the job. Sounded much more like a machine."', "29", COND_E, "Warmth & Engagement"),
    ('"Condition N was more robotic — you could hear it\nwent word by word instead of an entire sentence."', "25", COND_E, "Roboticness"),
    ('"It sounded more warm, more human.\nIt felt like I was heard better."', "25", COND_E, "Naturalness"),
    ('"The first version of John felt way more approachable,\nkind, and warm. I had to put more conscious effort\ninto interacting with the second version."', "86", COND_E, "Approachability"),
    ('"Condition E is very upbeat and spontaneous — a bit\ntoo extra and insincere, like a manager trying to\nmake the teambuilding day \'fun\'."', "44", COND_N, "Authenticity concern"),
    ('"The second version was more human-like,\nexcited, warm and enthusiastic."', "57", COND_E, "Humanness"),
    ('"Condition N felt like a synthetic voice trying to\nemulate a person doing a fake customer\nservice voice in a way I found unsettling."', "71", COND_E, "Roboticness"),
    ('"One used pauses like a human would;\nthe other just talked without significant breaks."', "75", "No preference", "Tone & Pacing"),
]

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor(BG)
fig.suptitle("Participant Voices: Selected Quotes from Open-Ended Responses",
             fontsize=14, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.08)

for idx, (quote, pid, pref, theme) in enumerate(QUOTES):
    ax = fig.add_subplot(gs[idx//2, idx%2])
    c = E_COLOR if pref==COND_E else (N_COLOR if pref==COND_N else NEUTRAL)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(2.2); spine.set_edgecolor(c)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((0,0),0.022,1,color=c,transform=ax.transAxes,clip_on=False,zorder=5))
    ax.text(0.06, 0.93, "“", fontsize=40, color=c, alpha=0.3, va="top", ha="left",
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.08, 0.78, quote, fontsize=9.5, va="top", ha="left",
            transform=ax.transAxes, linespacing=1.6, style="italic", color="#222222")
    ax.text(0.08, 0.09, f"— Participant {pid}", fontsize=8.5, va="bottom",
            ha="left", transform=ax.transAxes, color="#666666")
    ax.text(0.92, 0.09, f"Preferred: {pref}", fontsize=8, va="bottom", ha="right",
            transform=ax.transAxes, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=c, edgecolor="none"))

plt.savefig(OUT+"fig13_quotes.png", bbox_inches="tight"); plt.close()
print("✓ fig13_quotes.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 14 – Descriptor spectrum
# ─────────────────────────────────────────────────────────────────────────────
N_words = ["monotone","robotic","machine-like","formal","word-by-word",
           "unmotivated","bored","scripted","bossy","no pauses",
           "less rhythm","calmer","practical","hollow","mocking"]
E_words = ["warm","friendly","enthusiastic","upbeat","human-like",
           "natural","genuine","captivating","easy to listen to",
           "welcoming","excited","approachable","fun","expressive","engaged"]

fig, ax = plt.subplots(figsize=(15, 5.5))
ax.set_facecolor(BG); ax.set_xlim(-1,1); ax.set_ylim(-0.3,1.1); ax.axis("off")
arrow_y = 0.38
ax.annotate("", xy=(0.94,arrow_y), xytext=(-0.94,arrow_y),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.8))
ax.text(-0.94, arrow_y-0.13, f"{COND_N} described as...",
        ha="left", fontsize=10.5, color=N_COLOR, fontweight="bold")
ax.text(0.94, arrow_y-0.13, f"{COND_E} described as...",
        ha="right", fontsize=10.5, color=E_COLOR, fontweight="bold")
ax.axvline(0, color="#CCCCCC", lw=1, ls="--", ymin=0.05, ymax=0.95)
ax.text(0, arrow_y-0.13, "neutral", ha="center", fontsize=9, color=NEUTRAL)

np.random.seed(7)
for i, w in enumerate(N_words):
    xi = -0.96+(i%8)*0.12+np.random.uniform(-0.02,0.02)
    yi = 0.78-(i//8)*0.28+np.random.uniform(-0.03,0.03)
    ax.text(xi, yi, w, ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=N_COLOR, alpha=0.82, edgecolor="none"))

for i, w in enumerate(E_words):
    xi = 0.06+(i%8)*0.117+np.random.uniform(-0.02,0.02)
    yi = 0.78-(i//8)*0.28+np.random.uniform(-0.03,0.03)
    ax.text(xi, yi, w, ha="center", va="center", fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=E_COLOR, alpha=0.82, edgecolor="none"))

ax.set_title("Descriptive Language Used by Participants\n(Extracted and coded from open-ended responses)",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(OUT+"fig14_descriptor_spectrum.png"); plt.close()
print("✓ fig14_descriptor_spectrum.png")

print(f"\nAll 14 figures saved to: {OUT}")
