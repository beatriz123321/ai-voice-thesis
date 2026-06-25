import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap, math

N_COLOR  = "#4C72B0"
E_COLOR  = "#DD8452"
NEUTRAL  = "#8C8C8C"
BG       = "#FAFAFA"
COND_N, COND_E = "Condition N", "Condition E"

df = pd.read_excel("/Users/beatrizlopes/Downloads/full_responses_wide.xlsx")

def map_pref(row):
    vmap = {"Version A": COND_N, "Version B": COND_E}
    if "first"  in str(row["preferred_version"]).lower(): return vmap[row["S2S3_version"]]
    if "second" in str(row["preferred_version"]).lower(): return vmap[row["S4_version"]]
    return "No preference"
df["pref_actual"] = df.apply(map_pref, axis=1)

OUT  = "/Users/beatrizlopes/Downloads/thesis_v2/"
WRAP = 62

def get_responses(col, pref):
    sub = df[df["pref_actual"]==pref][["participant_id",col]].copy()
    sub = sub.dropna(subset=[col])
    sub = sub[sub[col].str.strip()!=""]
    return sub.reset_index(drop=True)

def count_lines(text, wrap=WRAP):
    lines = 0
    for para in str(text).split("\n"):
        lines += max(1, math.ceil(len(para)/wrap))
    return lines

def make_appendix_figure(col, title, filename):
    entries_n = get_responses(col, COND_N)
    entries_e = get_responses(col, COND_E)
    entries_x = get_responses(col, "No preference")

    LINE_H = 0.19; HEADER_H = 0.45; COL_PAD = 0.35

    def col_height(entries):
        h = COL_PAD
        for _, row in entries.iterrows():
            h += HEADER_H + count_lines(row[col])*LINE_H + 0.18
        return h

    hn = col_height(entries_n); he = col_height(entries_e)
    hx = col_height(entries_x) if len(entries_x)>0 else 0

    TITLE_H = 1.2; FOOTER_H = 0.4
    fig_h = TITLE_H + max(hn, he) + hx + FOOTER_H + 0.5
    fig_w = 18

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)

    title_ax = fig.add_axes([0, 1-TITLE_H/fig_h, 1, TITLE_H/fig_h])
    title_ax.set_facecolor("#2C3E50"); title_ax.axis("off")
    title_ax.text(0.5, 0.68, title, ha="center", va="center",
                  fontsize=15, fontweight="bold", color="white", transform=title_ax.transAxes)
    title_ax.text(0.5, 0.28, "Full participant responses — organised by condition preference  |  Appendix",
                  ha="center", va="center", fontsize=10, color="#AAAAAA", transform=title_ax.transAxes)

    content_top = 1-TITLE_H/fig_h
    header_band = 0.045

    hdr_ax = fig.add_axes([0, content_top-header_band, 1, header_band])
    hdr_ax.set_facecolor(BG); hdr_ax.axis("off")
    hdr_ax.add_patch(mpatches.FancyBboxPatch((0.01,0.05),0.47,0.85,
        boxstyle="round,pad=0.01", facecolor=N_COLOR, edgecolor="none",
        transform=hdr_ax.transAxes, clip_on=False))
    hdr_ax.add_patch(mpatches.FancyBboxPatch((0.52,0.05),0.47,0.85,
        boxstyle="round,pad=0.01", facecolor=E_COLOR, edgecolor="none",
        transform=hdr_ax.transAxes, clip_on=False))
    hdr_ax.text(0.245, 0.50, f"PREFERRED {COND_N.upper()}  (n = {len(entries_n)})",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", transform=hdr_ax.transAxes)
    hdr_ax.text(0.755, 0.50, f"PREFERRED {COND_E.upper()}  (n = {len(entries_e)})",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", transform=hdr_ax.transAxes)

    def draw_column(entries, color, x_left, x_right, top_y):
        cur_y = top_y
        for _, row in entries.iterrows():
            pid  = str(int(row["participant_id"])) if str(row["participant_id"]).replace(".","").isdigit() else str(row["participant_id"])
            text = str(row[col]).strip()
            wrapped = textwrap.fill(text, width=WRAP)
            n_lines = count_lines(text)

            card_header_frac = HEADER_H/fig_h
            card_text_frac   = (n_lines*LINE_H+0.12)/fig_h
            card_total       = card_header_frac+card_text_frac

            card_ax = fig.add_axes([x_left, cur_y-card_total, x_right-x_left, card_total])
            card_ax.set_facecolor("white")
            card_ax.set_xlim(0,1); card_ax.set_ylim(0,1); card_ax.axis("off")
            card_ax.add_patch(plt.Rectangle((0,0),0.014,1,color=color,
                              transform=card_ax.transAxes,clip_on=False))

            hf = card_header_frac/card_total
            card_ax.add_patch(plt.Rectangle((0.014,1-hf),0.986,hf,color=color,alpha=0.10,
                              transform=card_ax.transAxes,clip_on=False))
            card_ax.text(0.03,1-hf/2,f"Participant {pid}",
                         ha="left",va="center",fontsize=10,fontweight="bold",
                         color=color,transform=card_ax.transAxes)
            card_ax.text(0.03,(1-hf)*0.93,wrapped,
                         ha="left",va="top",fontsize=9.2,color="#222222",
                         transform=card_ax.transAxes,linespacing=1.5,style="italic")
            for spine in card_ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(0.6); spine.set_edgecolor("#DDDDDD")

            cur_y -= card_total + 0.008/fig_h*2
        return cur_y

    body_top = content_top - header_band - 0.01
    draw_column(entries_n, N_COLOR, 0.01, 0.49, body_top)
    draw_column(entries_e, E_COLOR, 0.51, 0.99, body_top)

    if len(entries_x) > 0:
        no_pref_top = body_top - max(hn, he)/fig_h - 0.01
        sep_ax = fig.add_axes([0.01, no_pref_top-0.025, 0.98, 0.022])
        sep_ax.set_facecolor(NEUTRAL); sep_ax.axis("off")
        sep_ax.text(0.5,0.5,f"NO PREFERENCE  (n = {len(entries_x)})",
                    ha="center",va="center",fontsize=11,fontweight="bold",
                    color="white",transform=sep_ax.transAxes)
        draw_column(entries_x, NEUTRAL, 0.01, 0.49, no_pref_top-0.05)

    plt.savefig(OUT+filename, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"✓ {filename}")

make_appendix_figure(
    col      = "difference_description",
    title    = "Appendix A — What Differences Did You Notice Between the Two Conditions?",
    filename = "appendix_A_difference_descriptions.png",
)
make_appendix_figure(
    col      = "preference_reason",
    title    = "Appendix B — Why Did You Prefer That Condition?",
    filename = "appendix_B_preference_reasons.png",
)

print(f"\nBoth appendix figures saved to: {OUT}")
