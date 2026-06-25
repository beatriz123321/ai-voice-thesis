import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy import stats

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

PDF_PATH = '/Users/beatrizlopes/Data-Systems-Project---UvA-Impact-Story-Generation-3/survey_results.pdf'

# ── DATA ──────────────────────────────────────────────────────────────────────

gender      = ['F','F','F','F','F','M','M','F','M','M','F','F','M','M','F','M','F']
age         = [57,26,24,22,24,26,54,28,26,23,23,24,27,29,67,67,40]
location    = ['Lisbon, PT','Lisbon, PT','Lisbon, PT','Amsterdam, NL','Madrid, ES',
               'Madrid, ES','Lisbon, PT','Lisbon, PT','Amsterdam, NL','Amsterdam, NL',
               'Amsterdam, NL','Eindhoven, NL','Amsterdam, NL','Amsterdam, NL',
               'Noord-Brabant, NL','Noord-Brabant, NL','Lisbon, PT']
english_lvl = ['C1/C2']*17
english_lvl[9]  = 'Native'
english_lvl[16] = 'B1/B2'
listen_dev  = ['In-ear','Loudspeakers','Loudspeakers','Loudspeakers','In-ear','Loudspeakers',
               'Loudspeakers','Loudspeakers','In-ear','Over-ear','Loudspeakers','In-ear',
               'Over-ear','In-ear','Loudspeakers','Loudspeakers','Unknown']

# Female voice ratings (Kristen, Christina, Hannah, Jessica)
kristen  = [7,9,9,7,3,7,4,7,3,6,4,6,8,6,8,7,8]
christina= [9,7,7,5,8,8,5,6,8,8,7,6,9,5,7,6,8]
hannah   = [7,9,10,6,3,1,3,3,2,3,1,7,7,4,6,8,8]
jessica  = [4,8,6,6,5,5,2,4,5,5,5,7,3,6,7,8,7]

# Male voice ratings (Henry, Mark, John, Brady)
henry = [5,5,8,7,7,6,5,6,3,4,4,8,5,4,7,7,5]
mark  = [4,6,10,3,9,8,2,7,5,7,5,7,2,7,5,5,7]
john  = [8,4,9,4,4,4,1,6,7,5,7,7,5,5,8,6,8]
brady = [9,8,8,6,5,5,6,5,8,4,5,6,4,6,7,8,9]

# Female rankings: [Kristen, Christina, Hannah, Jessica] — 1=best
fem_rank = [
    [2,1,3,4],[1,4,2,3],[2,1,3,4],[1,4,3,2],[4,1,3,2],
    [3,1,4,2],[2,1,3,4],[1,2,4,3],[2,1,4,3],[2,1,4,3],
    [3,1,4,2],[3,4,2,1],[2,1,3,4],[1,3,4,2],[1,3,4,2],
    [3,4,2,1],[4,3,2,1],
]

# Male rankings: [Henry, Mark, John, Brady] — 1=best
mal_rank = [
    [3,4,2,1],[3,2,4,1],[2,1,4,3],[1,4,3,2],[2,1,4,3],
    [3,1,2,4],[2,3,4,1],[2,1,3,4],[4,3,2,1],[2,1,4,3],
    [4,2,1,3],[1,3,2,4],[2,3,1,4],[2,1,3,4],[2,4,1,3],
    [2,4,3,1],[4,3,2,1],
]

# Factors (unique participants who mentioned each, across all pairwise Qs)
factors = {
    'Tone / Warmth':     14,
    'Naturalness':       13,
    'Overall Pleasantness': 13,
    'Pace':              12,
    'Clarity':            6,
    'Gender':             5,
    'Other':              4,
}

# Pairwise block winners (Block 1, Block 2)
block1 = ['John','Kristen','Christina','Brady','Christina','John','Henry',
          'Kristen','John','Christina','Mark','John','Christina','Henry',
          'Kristen','Jessica','Jessica']
block2 = ['Brady','Hannah','Mark','Jessica','Mark','Mark','Brady',
          'Christina','Brady','Kristen','John','Henry','Kristen','Mark',
          'John','Brady','Hannah']

# ── COLOUR PALETTES ───────────────────────────────────────────────────────────
FEM_COLORS = ['#E07B9A','#A855C8','#F5A623','#4AB8C1']   # Kristen, Christina, Hannah, Jessica
MAL_COLORS = ['#4A90D9','#E84040','#2ECC71','#F39C12']   # Henry, Mark, John, Brady

ACCENT = '#5B4FCF'
SOFT_BG = '#F7F7F9'

# ── HELPERS ───────────────────────────────────────────────────────────────────

def rank_to_points(rank_lists):
    """Convert rank lists (1=best) to points (4=best) per voice."""
    arr = np.array(rank_lists)  # shape (17, 4)
    n_voices = arr.shape[1]
    return (n_voices + 1) - arr   # 1→4, 2→3, 3→2, 4→1

def first_place_counts(rank_lists):
    arr = np.array(rank_lists)
    return [(arr[:, i] == 1).sum() for i in range(arr.shape[1])]

def mean_sd(data):
    a = np.array(data)
    return a.mean(), a.std(ddof=1)

# ── PAGE 1 · DEMOGRAPHICS ─────────────────────────────────────────────────────

def page_demographics(pdf):
    fig = plt.figure(figsize=(12, 9), facecolor=SOFT_BG)
    fig.suptitle('Participant Demographics  (N = 17)', fontsize=16, fontweight='bold', y=0.97)

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.45)

    # 1 · Gender
    ax1 = fig.add_subplot(gs[0, 0])
    g_counts = [gender.count('F'), gender.count('M')]
    wedges, texts, autotexts = ax1.pie(
        g_counts, labels=['Female (10)', 'Male (7)'],
        colors=['#E07B9A','#4A90D9'], autopct='%1.0f%%',
        startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Gender', fontweight='bold')

    # 2 · Age distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(age, bins=[18,30,40,50,60,70,80], color=ACCENT, edgecolor='white', rwidth=0.85)
    ax2.set_xlabel('Age (years)')
    ax2.set_ylabel('Count')
    ax2.set_title('Age Distribution', fontweight='bold')
    ax2.set_xticks([18,30,40,50,60,70,80])
    ax2.set_facecolor(SOFT_BG)
    mu, sd = mean_sd(age)
    ax2.axvline(mu, color='red', linestyle='--', linewidth=1.2, label=f'Mean = {mu:.1f}')
    ax2.legend(fontsize=8)

    # 3 · English proficiency
    ax3 = fig.add_subplot(gs[0, 2])
    lvl_counts = {'C1/C2': english_lvl.count('C1/C2'),
                  'Native': english_lvl.count('Native'),
                  'B1/B2':  english_lvl.count('B1/B2')}
    bars = ax3.bar(lvl_counts.keys(), lvl_counts.values(),
                   color=['#A855C8','#2ECC71','#F5A623'], edgecolor='white')
    ax3.set_ylabel('Count')
    ax3.set_title('English Proficiency', fontweight='bold')
    ax3.set_facecolor(SOFT_BG)
    for b in bars:
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                 str(int(b.get_height())), ha='center', va='bottom', fontsize=10)

    # 4 · Listening device
    ax4 = fig.add_subplot(gs[1, 0])
    dev_counts = {}
    for d in listen_dev:
        dev_counts[d] = dev_counts.get(d, 0) + 1
    ax4.barh(list(dev_counts.keys()), list(dev_counts.values()),
             color=ACCENT, edgecolor='white')
    ax4.set_xlabel('Count')
    ax4.set_title('Listening Device', fontweight='bold')
    ax4.set_facecolor(SOFT_BG)
    for i, v in enumerate(dev_counts.values()):
        ax4.text(v + 0.05, i, str(v), va='center', fontsize=9)

    # 5 · Location
    ax5 = fig.add_subplot(gs[1, 1:])
    loc_counts = {}
    for l in location:
        loc_counts[l] = loc_counts.get(l, 0) + 1
    sorted_locs = sorted(loc_counts.items(), key=lambda x: -x[1])
    names, cnts = zip(*sorted_locs)
    colors_loc = ['#4AB8C1','#E07B9A','#F5A623','#A855C8','#4A90D9']
    ax5.bar(names, cnts, color=colors_loc[:len(names)], edgecolor='white')
    ax5.set_ylabel('Count')
    ax5.set_title('Location', fontweight='bold')
    ax5.set_facecolor(SOFT_BG)
    ax5.tick_params(axis='x', rotation=20)
    for i, v in enumerate(cnts):
        ax5.text(i, v + 0.05, str(v), ha='center', va='bottom', fontsize=9)

    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── PAGE 2 · FEMALE VOICE RATINGS ─────────────────────────────────────────────

def page_female_ratings(pdf):
    fig = plt.figure(figsize=(14, 10), facecolor=SOFT_BG)
    fig.suptitle('Female Voice Appeal Ratings (1–10)\nKristen · Christina · Hannah · Jessica',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    voices_f = ['Kristen', 'Christina', 'Hannah', 'Jessica']
    data_f   = [kristen, christina, hannah, jessica]
    means_f  = [np.mean(d) for d in data_f]
    sds_f    = [np.std(d, ddof=1) for d in data_f]

    # A · Bar chart with error bars
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(voices_f, means_f, color=FEM_COLORS, edgecolor='white')
    ax1.errorbar(voices_f, means_f, yerr=sds_f, fmt='none', color='#444', capsize=6, linewidth=1.5)
    ax1.set_ylim(0, 12)
    ax1.set_ylabel('Mean Rating ± SD')
    ax1.set_title('Mean Rating per Voice', fontweight='bold')
    ax1.set_facecolor(SOFT_BG)
    for i, (m, sd) in enumerate(zip(means_f, sds_f)):
        ax1.text(i, m + sd + 0.35, f'{m:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # B · Box plot
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([kristen, christina, hannah, jessica], tick_labels=voices_f,
                     patch_artist=True, medianprops={'color':'black','linewidth':2})
    for patch, color in zip(bp['boxes'], FEM_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylim(0, 11)
    ax2.set_ylabel('Rating')
    ax2.set_title('Rating Distribution (Box Plot)', fontweight='bold')
    ax2.set_facecolor(SOFT_BG)

    # C · Individual ratings heatmap
    ax3 = fig.add_subplot(gs[1, :])
    heat_data = np.array([kristen, christina, hannah, jessica])  # (4, 17)
    im = ax3.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=1, vmax=10)
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(voices_f)
    ax3.set_xticks(range(17))
    ax3.set_xticklabels([f'P{i+1}' for i in range(17)], fontsize=8)
    ax3.set_xlabel('Participant')
    ax3.set_title('Individual Ratings Heatmap (1=Low, 10=High)', fontweight='bold')
    for r in range(4):
        for c in range(17):
            ax3.text(c, r, str(heat_data[r, c]), ha='center', va='center',
                     fontsize=7.5, color='black' if 3 < heat_data[r, c] < 8 else 'white')
    plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.12, fraction=0.03)

    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── PAGE 3 · MALE VOICE RATINGS ───────────────────────────────────────────────

def page_male_ratings(pdf):
    fig = plt.figure(figsize=(14, 10), facecolor=SOFT_BG)
    fig.suptitle('Male Voice Appeal Ratings (1–10)\nHenry · Mark · John · Brady',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    voices_m = ['Henry', 'Mark', 'John', 'Brady']
    data_m   = [henry, mark, john, brady]
    means_m  = [np.mean(d) for d in data_m]
    sds_m    = [np.std(d, ddof=1) for d in data_m]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(voices_m, means_m, color=MAL_COLORS, edgecolor='white')
    ax1.errorbar(voices_m, means_m, yerr=sds_m, fmt='none', color='#444', capsize=6, linewidth=1.5)
    ax1.set_ylim(0, 12)
    ax1.set_ylabel('Mean Rating ± SD')
    ax1.set_title('Mean Rating per Voice', fontweight='bold')
    ax1.set_facecolor(SOFT_BG)
    for i, (m, sd) in enumerate(zip(means_m, sds_m)):
        ax1.text(i, m + sd + 0.35, f'{m:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(data_m, tick_labels=voices_m, patch_artist=True,
                     medianprops={'color':'black','linewidth':2})
    for patch, color in zip(bp['boxes'], MAL_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylim(0, 11)
    ax2.set_ylabel('Rating')
    ax2.set_title('Rating Distribution (Box Plot)', fontweight='bold')
    ax2.set_facecolor(SOFT_BG)

    ax3 = fig.add_subplot(gs[1, :])
    heat_data = np.array([henry, mark, john, brady])
    im = ax3.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=1, vmax=10)
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(voices_m)
    ax3.set_xticks(range(17))
    ax3.set_xticklabels([f'P{i+1}' for i in range(17)], fontsize=8)
    ax3.set_xlabel('Participant')
    ax3.set_title('Individual Ratings Heatmap (1=Low, 10=High)', fontweight='bold')
    for r in range(4):
        for c in range(17):
            ax3.text(c, r, str(heat_data[r, c]), ha='center', va='center',
                     fontsize=7.5, color='black' if 3 < heat_data[r, c] < 8 else 'white')
    plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.12, fraction=0.03)

    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── PAGE 4 · ALL 8 VOICES COMPARISON ──────────────────────────────────────────

def page_all_voices(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=SOFT_BG)
    fig.suptitle('All 8 Voices — Mean Appeal Ratings Compared', fontsize=15, fontweight='bold')

    all_voices = ['Kristen','Christina','Hannah','Jessica','Henry','Mark','John','Brady']
    all_data   = [kristen, christina, hannah, jessica, henry, mark, john, brady]
    all_means  = [np.mean(d) for d in all_data]
    all_sds    = [np.std(d, ddof=1) for d in all_data]
    all_colors = FEM_COLORS + MAL_COLORS

    # Left: grouped by gender
    ax = axes[0]
    x = np.arange(8)
    bars = ax.bar(x, all_means, color=all_colors, edgecolor='white', width=0.65)
    ax.errorbar(x, all_means, yerr=all_sds, fmt='none', color='#444', capsize=5, linewidth=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(all_voices, rotation=20, ha='right')
    ax.set_ylim(0, 11)
    ax.set_ylabel('Mean Rating ± SD')
    ax.set_title('Mean Ratings — All Voices', fontweight='bold')
    ax.set_facecolor(SOFT_BG)
    ax.axvline(3.5, color='#999', linestyle='--', linewidth=1)
    ax.text(1.5, 10.5, 'Female voices', ha='center', fontsize=9, color='#666')
    ax.text(5.5, 10.5, 'Male voices',   ha='center', fontsize=9, color='#666')
    for i, (m, sd) in enumerate(zip(all_means, all_sds)):
        ax.text(i, m + sd + 0.25, f'{m:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Right: ranked horizontal bar
    ax2 = axes[1]
    order = np.argsort(all_means)[::-1]
    ranked_names   = [all_voices[i] for i in order]
    ranked_means   = [all_means[i] for i in order]
    ranked_sds     = [all_sds[i] for i in order]
    ranked_colors  = [all_colors[i] for i in order]
    y = np.arange(len(order))
    ax2.barh(y, ranked_means, xerr=ranked_sds, color=ranked_colors, edgecolor='white',
             capsize=4, error_kw={'linewidth':1.2})
    ax2.set_yticks(y)
    ax2.set_yticklabels(ranked_names)
    ax2.set_xlabel('Mean Rating')
    ax2.set_title('Voices Ranked by Mean Rating', fontweight='bold')
    ax2.set_facecolor(SOFT_BG)
    ax2.set_xlim(0, 11)
    for i, (m, sd) in enumerate(zip(ranked_means, ranked_sds)):
        ax2.text(m + sd + 0.2, i, f'{m:.2f}', va='center', fontsize=8, fontweight='bold')

    fem_patch = mpatches.Patch(color='#E07B9A', label='Female voices')
    mal_patch = mpatches.Patch(color='#4A90D9', label='Male voices')
    fig.legend(handles=[fem_patch, mal_patch], loc='lower center', ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.set_facecolor(SOFT_BG)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── PAGE 5 · RANKINGS ─────────────────────────────────────────────────────────

def page_rankings(pdf):
    fig = plt.figure(figsize=(14, 9), facecolor=SOFT_BG)
    fig.suptitle('Voice Preference Rankings', fontsize=15, fontweight='bold', y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    voices_f = ['Kristen','Christina','Hannah','Jessica']
    voices_m = ['Henry','Mark','John','Brady']

    # A · Female 1st-place counts
    ax1 = fig.add_subplot(gs[0, 0])
    fp_f = first_place_counts(fem_rank)
    bars = ax1.bar(voices_f, fp_f, color=FEM_COLORS, edgecolor='white')
    ax1.set_ylabel('# Participants ranking 1st')
    ax1.set_title('Female Voices — 1st Place Count', fontweight='bold')
    ax1.set_facecolor(SOFT_BG)
    ax1.set_ylim(0, 12)
    for b, v in zip(bars, fp_f):
        ax1.text(b.get_x()+b.get_width()/2, v+0.1, str(v), ha='center', va='bottom', fontweight='bold')

    # B · Male 1st-place counts
    ax2 = fig.add_subplot(gs[0, 1])
    fp_m = first_place_counts(mal_rank)
    bars = ax2.bar(voices_m, fp_m, color=MAL_COLORS, edgecolor='white')
    ax2.set_ylabel('# Participants ranking 1st')
    ax2.set_title('Male Voices — 1st Place Count', fontweight='bold')
    ax2.set_facecolor(SOFT_BG)
    ax2.set_ylim(0, 12)
    for b, v in zip(bars, fp_m):
        ax2.text(b.get_x()+b.get_width()/2, v+0.1, str(v), ha='center', va='bottom', fontweight='bold')

    # C · Female stacked rank distribution
    ax3 = fig.add_subplot(gs[1, 0])
    rank_arr_f = np.array(fem_rank)
    bottom = np.zeros(4)
    rank_colors = ['#2ECC71','#A8D8A8','#F5A623','#E84040']
    rank_labels = ['1st','2nd','3rd','4th']
    for rank_val, col, lbl in zip([1,2,3,4], rank_colors, rank_labels):
        counts = [(rank_arr_f[:, i] == rank_val).sum() for i in range(4)]
        ax3.bar(voices_f, counts, bottom=bottom, color=col, label=lbl, edgecolor='white')
        bottom += np.array(counts)
    ax3.set_ylabel('Number of participants')
    ax3.set_title('Female — Full Rank Distribution', fontweight='bold')
    ax3.set_facecolor(SOFT_BG)
    ax3.legend(loc='upper right', fontsize=8)

    # D · Male stacked rank distribution
    ax4 = fig.add_subplot(gs[1, 1])
    rank_arr_m = np.array(mal_rank)
    bottom = np.zeros(4)
    for rank_val, col, lbl in zip([1,2,3,4], rank_colors, rank_labels):
        counts = [(rank_arr_m[:, i] == rank_val).sum() for i in range(4)]
        ax4.bar(voices_m, counts, bottom=bottom, color=col, label=lbl, edgecolor='white')
        bottom += np.array(counts)
    ax4.set_ylabel('Number of participants')
    ax4.set_title('Male — Full Rank Distribution', fontweight='bold')
    ax4.set_facecolor(SOFT_BG)
    ax4.legend(loc='upper right', fontsize=8)

    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── PAGE 6 · FACTORS INFLUENCING PREFERENCE ───────────────────────────────────

def page_factors(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=SOFT_BG)
    fig.suptitle('Factors Influencing Voice Preference', fontsize=15, fontweight='bold')

    fac_names = list(factors.keys())
    fac_vals  = list(factors.values())
    order     = np.argsort(fac_vals)[::-1]
    sorted_names = [fac_names[i] for i in order]
    sorted_vals  = [fac_vals[i]  for i in order]
    pct = [v / 17 * 100 for v in sorted_vals]
    colors_fac = sns.color_palette('husl', len(fac_names))
    sorted_colors = [colors_fac[i] for i in order]

    ax1 = axes[0]
    bars = ax1.barh(sorted_names, sorted_vals, color=sorted_colors, edgecolor='white')
    ax1.set_xlabel('Number of participants mentioning factor')
    ax1.set_title('Factor Frequency (N=17)', fontweight='bold')
    ax1.set_facecolor(SOFT_BG)
    ax1.set_xlim(0, 19)
    ax1.axvline(17, color='#999', linestyle='--', linewidth=0.8)
    ax1.text(17.1, -0.7, 'N=17', fontsize=7, color='#999')
    for b, v, p in zip(bars, sorted_vals, pct):
        ax1.text(v + 0.2, b.get_y() + b.get_height()/2,
                 f'{v} ({p:.0f}%)', va='center', fontsize=9, fontweight='bold')

    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        sorted_vals, labels=sorted_names, colors=sorted_colors,
        autopct='%1.0f%%', startangle=140,
        textprops={'fontsize': 8}, pctdistance=0.78)
    ax2.set_title('Relative Factor Importance', fontweight='bold')

    fig.set_facecolor(SOFT_BG)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── PAGE 7 · PAIRWISE COMPARISON WINNERS ──────────────────────────────────────

def page_pairwise(pdf):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=SOFT_BG)
    fig.suptitle('Pairwise Comparison Winners (Tournament Blocks)', fontsize=15, fontweight='bold')

    all_voice_colors = {
        'Kristen':'#E07B9A','Christina':'#A855C8','Hannah':'#F5A623','Jessica':'#4AB8C1',
        'Henry':'#4A90D9','Mark':'#E84040','John':'#2ECC71','Brady':'#F39C12'
    }

    def count_winners(block):
        from collections import Counter
        c = Counter(block)
        names = sorted(c.keys(), key=lambda x: -c[x])
        vals  = [c[n] for n in names]
        cols  = [all_voice_colors.get(n, '#999') for n in names]
        return names, vals, cols

    for ax, block, title in zip(
            axes[:2], [block1, block2], ['Block 1 Winners', 'Block 2 Winners']):
        names, vals, cols = count_winners(block)
        bars = ax.bar(names, vals, color=cols, edgecolor='white')
        ax.set_ylabel('Count')
        ax.set_title(title, fontweight='bold')
        ax.set_facecolor(SOFT_BG)
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(0, 7)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.05, str(v), ha='center', va='bottom', fontweight='bold')

    # Combined
    from collections import Counter
    combined = block1 + block2
    c = Counter(combined)
    names = sorted(c.keys(), key=lambda x: -c[x])
    vals  = [c[n] for n in names]
    cols  = [all_voice_colors.get(n, '#999') for n in names]
    ax3 = axes[2]
    bars = ax3.bar(names, vals, color=cols, edgecolor='white')
    ax3.set_ylabel('Total wins')
    ax3.set_title('Combined Block Wins (Block 1 + 2)', fontweight='bold')
    ax3.set_facecolor(SOFT_BG)
    ax3.tick_params(axis='x', rotation=30)
    ax3.set_ylim(0, 9)
    for b, v in zip(bars, vals):
        ax3.text(b.get_x()+b.get_width()/2, v+0.05, str(v), ha='center', va='bottom', fontweight='bold')

    # Legend showing which voices are female/male
    fem_patch = mpatches.Patch(color='#E07B9A', label='Female voices (warm tones)')
    mal_patch = mpatches.Patch(color='#4A90D9', label='Male voices (cool tones)')
    fig.legend(handles=[fem_patch, mal_patch], loc='lower center', ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.04))

    fig.set_facecolor(SOFT_BG)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── PAGE 8 · INDIVIDUAL PROFILES ──────────────────────────────────────────────

def page_profiles(pdf):
    """Radar-style overview: each participant's top-rated female and male voice."""
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=SOFT_BG)
    fig.suptitle('Per-Participant Top Voice Choices', fontsize=15, fontweight='bold')

    voices_f = ['Kristen','Christina','Hannah','Jessica']
    voices_m = ['Henry','Mark','John','Brady']
    data_f = [kristen, christina, hannah, jessica]
    data_m = [henry, mark, john, brady]

    x = np.arange(17)
    width = 0.35
    best_f_idx = [np.argmin([fem_rank[p][v] for v in range(4)]) for p in range(17)]
    best_m_idx = [np.argmin([mal_rank[p][v] for v in range(4)]) for p in range(17)]

    best_f_ratings = [data_f[best_f_idx[p]][p] for p in range(17)]
    best_m_ratings = [data_m[best_m_idx[p]][p] for p in range(17)]
    best_f_names   = [voices_f[i] for i in best_f_idx]
    best_m_names   = [voices_m[i] for i in best_m_idx]
    best_f_colors  = [FEM_COLORS[i] for i in best_f_idx]
    best_m_colors  = [MAL_COLORS[i] for i in best_m_idx]

    for i in range(17):
        ax.bar(i - width/2, best_f_ratings[i], width, color=best_f_colors[i], edgecolor='white', alpha=0.85)
        ax.bar(i + width/2, best_m_ratings[i], width, color=best_m_colors[i], edgecolor='white', alpha=0.85)
        ax.text(i - width/2, best_f_ratings[i] + 0.1, best_f_names[i][0], ha='center', fontsize=6.5, color='#333')
        ax.text(i + width/2, best_m_ratings[i] + 0.1, best_m_names[i][0], ha='center', fontsize=6.5, color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}\n{gender[i]}·{age[i]}' for i in range(17)], fontsize=7.5)
    ax.set_ylabel('Rating given to top-ranked voice')
    ax.set_ylim(0, 12)
    ax.set_title('Each participant\'s #1 ranked female (left bar) and #1 male (right bar)',
                 fontweight='bold', fontsize=10)
    ax.set_facecolor(SOFT_BG)

    # Legend for voice names
    legend_handles = ([mpatches.Patch(color=c, label=v) for v, c in zip(voices_f, FEM_COLORS)] +
                      [mpatches.Patch(color=c, label=v) for v, c in zip(voices_m, MAL_COLORS)])
    ax.legend(handles=legend_handles, ncol=4, loc='upper right', fontsize=7.5, frameon=True)

    note = 'Letter on bar = first letter of voice name. Gender·Age shown on x-axis.'
    fig.text(0.5, 0.01, note, ha='center', fontsize=8, color='#888')

    fig.set_facecolor(SOFT_BG)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── PAGE 9 · SUMMARY STATISTICS TABLE ─────────────────────────────────────────

def page_summary(pdf):
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=SOFT_BG)
    fig.suptitle('Summary Statistics — All 8 Voices', fontsize=15, fontweight='bold')
    ax.axis('off')

    voices_all = ['Kristen','Christina','Hannah','Jessica','Henry','Mark','John','Brady']
    data_all   = [kristen, christina, hannah, jessica, henry, mark, john, brady]
    gnd        = ['F','F','F','F','M','M','M','M']

    rows = []
    for v, d, g in zip(voices_all, data_all, gnd):
        a = np.array(d)
        rows.append([v, g, f'{a.mean():.2f}', f'{a.std(ddof=1):.2f}',
                     f'{np.median(a):.1f}', str(int(a.min())), str(int(a.max())),
                     str((a >= 7).sum()), str((a <= 3).sum())])

    col_labels = ['Voice','Gender','Mean','SD','Median','Min','Max','Ratings ≥7','Ratings ≤3']
    colors_rows = []
    for i, (v, g) in enumerate(zip(voices_all, gnd)):
        c = FEM_COLORS[i] if g == 'F' else MAL_COLORS[i-4]
        colors_rows.append([c] + ['#F0F0F0']*8)

    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellColours=colors_rows,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#333333')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('N = 17 participants per voice', fontsize=10, pad=20)

    # Best voice annotation
    best_f = voices_all[:4][np.argmax([np.mean(d) for d in data_all[:4]])]
    best_m = voices_all[4:][np.argmax([np.mean(d) for d in data_all[4:]])]
    fig.text(0.5, 0.04,
             f'Top female voice by mean rating: {best_f}  |  Top male voice by mean rating: {best_m}',
             ha='center', fontsize=11, fontweight='bold', color=ACCENT)

    fig.set_facecolor(SOFT_BG)
    plt.savefig(pdf, format='pdf', bbox_inches='tight', facecolor=SOFT_BG)
    plt.close()

# ── RUN ───────────────────────────────────────────────────────────────────────

with PdfPages(PDF_PATH) as pdf:
    page_demographics(pdf)
    page_female_ratings(pdf)
    page_male_ratings(pdf)
    page_all_voices(pdf)
    page_rankings(pdf)
    page_factors(pdf)
    page_pairwise(pdf)
    page_profiles(pdf)
    page_summary(pdf)

    d = pdf.infodict()
    d['Title']   = '1st Experiment Survey Results'
    d['Author']  = 'Beatriz Lopes — UvA Impact Story Generation'
    d['Subject'] = 'TTS Voice Preference Analysis (N=17)'

print(f'Done → {PDF_PATH}')
