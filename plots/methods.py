"""Figures 1, 2, 4, 6–8 and 10 from the methods chapter."""
from collections import defaultdict
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as DrawingPath
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from src.config import T_START, T_END, PHASES
from src.cooling_generator import F4_SETTINGS, smooth_stages
from src.curves import curve_nodes, time_at_temperature, temperature_at
from src.features import prepare_features, reconstruct_curve, temperature_grid, time_grid


NAMES = ("Ferryt", "Perlit", "Bainit", "Martenzyt")
COLORS = ("#4477aa", "#dd9944", "#228866", "#bb5566")
ZERO_TOL_PCT = 1e-8
FAMILIES = {
    "F1": "stała szybkość",
    "F2": "zmiana szybkości",
    "F3": "postój izotermiczny",
    "F4": "malejąca szybkość",
}


def family(row):
    return "F3" if row["family"] == "sensitivity" else row["family"].split("_")[0]


def labels(rows):
    return np.array([[float(r[p]) for p in PHASES] for r in rows]) * 100


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", dpi=200, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def sample_curves(rows, metadata, count, rng):
    """Próbka proporcjonalna do bloków; zachowuje oba skrajne czasy rodziny."""
    count = min(count, len(rows))
    if count == len(rows):
        return rows
    extremes = {r["curve_id"] for r in (
        min(rows, key=lambda r: float(r["duration_s"])),
        max(rows, key=lambda r: float(r["duration_s"])))}
    chosen = [r for r in rows if r["curve_id"] in extremes]
    groups = defaultdict(list)
    for row in rows:
        if row["curve_id"] not in extremes:
            band = metadata[row["curve_id"]]["params"].get("hold_band", "")
            groups[(row["family"], str(band))].append(row)
    groups = [groups[k] for k in sorted(groups)]
    remaining = count - len(chosen)
    if remaining < len(groups):
        raise ValueError("Zbyt mała próbka, by uwzględnić każdy blok")
    sizes = np.array([len(g) for g in groups])
    ideal = (remaining - len(groups)) * (sizes - 1) / (sizes - 1).sum()
    quotas = 1 + np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal - np.floor(ideal)), kind="stable")[:remaining - quotas.sum()]:
        quotas[i] += 1
    for group, quota in zip(groups, quotas):
        chosen.extend(group[i] for i in rng.choice(len(group), int(quota), replace=False))
    assert len(chosen) == count and len({r["curve_id"] for r in chosen}) == count
    return sorted(chosen, key=lambda r: int(r["curve_id"]))


def curve_overview(rows, metadata, out, seed, per_family):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True, sharey=True)
    rng = np.random.default_rng(seed)
    for ax, (name, description), color in zip(axes.flat, FAMILIES.items(), COLORS):
        subset = [r for r in rows if family(r) == name]
        shown = sample_curves(subset, metadata, per_family, rng)
        for row in shown:
            t, temperature = curve_nodes(row["stages"])
            # Gęste punkty zapobiegają rysowaniu prostych odcinków w log(t).
            # Węzły etapów zachowują dokładne miejsca zmiany szybkości i postojów.
            query = np.unique(np.r_[np.geomspace(0.01, t[-1], 600), t[t >= 0.01]])
            ax.plot(query, np.interp(query, t, temperature), color=color,
                    alpha=0.20, linewidth=0.6)
        ax.set(xscale="log", xlim=(0.01, 120000), ylim=(T_END, T_START),
               yticks=[20, 200, 400, 600, 830])
        ax.set_title(f"{name}: {description}\n{len(shown)} z {len(subset)} krzywych", fontsize=10)
        ax.set_xticks([0.01, 1, 100, 10000])
        ax.grid(alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperatura [°C]")
    for ax in axes[-1, :]:
        ax.set_xlabel("Czas od początku chłodzenia [s]")
    fig.tight_layout(h_pad=1.8, w_pad=1.8)
    save(fig, out, "przebiegi_rodzin")


def histograms(rows, out):
    fig, axes = plt.subplots(4, 4, figsize=(8.4, 7.0), sharex=True, sharey=True)
    bins = np.arange(0, 101, 5)
    for i, name in enumerate(FAMILIES):
        subset = [r for r in rows if family(r) == name]
        y = labels(subset)
        zero_pct = 100 * (np.abs(y) <= 1e-8).mean(axis=0)
        for j, (phase, color) in enumerate(zip(PHASES, COLORS)):
            ax = axes[i, j]
            # Jedynie błędy zmiennoprzecinkowe (np. M = -1e-14%) są zerowane.
            counts, _ = np.histogram(np.clip(y[:, j], 0, 100), bins=bins)
            assert counts.sum() == len(subset)
            pct = 100 * counts / len(subset)
            ax.bar(bins[:-1], pct, width=5, align="edge", color=color,
                   edgecolor="white", linewidth=0.35, alpha=0.85)
            zero = f"{zero_pct[j]:.1f}"
            ax.text(0.04 if j == 0 else 0.96, 0.94,
                    f"Udział zerowy:\n{zero}% krzywych", transform=ax.transAxes,
                    ha="left" if j == 0 else "right", va="top", fontsize=7.8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1})
            ax.set(xlim=(0, 100), ylim=(0, 105),
                   xticks=np.arange(0, 101, 25), yticks=np.arange(0, 101, 20))
            ax.grid(axis="y", alpha=0.18)
            if i == 0:
                ax.set_title(NAMES[j], color=color, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{name}\n$n={len(subset)}$", fontsize=9)
    fig.supxlabel("Końcowy udział składnika [%]", fontsize=10)
    fig.supylabel("Odsetek krzywych w rodzinie [%]", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 1), h_pad=1, w_pad=1)
    save(fig, out, "histogramy_udzialow")


def family_phase_summary(rows, out):
    """Średnie udziały w każdej rodzinie krzywych."""
    means, counts = [], []
    for name in FAMILIES:
        subset = [row for row in rows if family(row) == name]
        if not subset:
            raise ValueError(f"Brak krzywych rodziny {name}")
        y = labels(subset)
        mean = y.mean(axis=0)
        np.testing.assert_allclose(mean.sum(), 100, atol=1e-8, rtol=0)
        means.append(mean)
        counts.append(len(subset))
    means = np.array(means)
    positions = np.arange(len(FAMILIES))
    row_labels = [f"{name} ($n={n}$)" for name, n in zip(FAMILIES, counts)]

    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    left = np.zeros(len(FAMILIES))
    for j, (phase, name, color) in enumerate(zip(PHASES, NAMES, COLORS)):
        values = means[:, j]
        ax.barh(positions, values, left=left, height=.57, color=color,
                edgecolor="white", linewidth=.6, label=f"{name} ({phase})")
        for row, value in enumerate(values):
            if value <= ZERO_TOL_PCT:
                continue
            center = left[row] + value / 2
            label = f"{value:.1f}"
            if value >= 4:
                ax.text(center, row, label, ha="center", va="center", fontsize=9,
                        color="black" if phase == "P" else "white")
            else:
                ax.annotate(label, (center, row-.285), xytext=(center, row-.43),
                            ha="center", va="bottom", fontsize=8,
                            arrowprops=dict(arrowstyle="-", lw=.6, color="0.4"))
        left += values
    ax.set(yticks=positions, yticklabels=row_labels, xlim=(0, 100),
           ylim=(len(FAMILIES)-.5, -.65), xticks=np.arange(0, 101, 20),
           xlabel="Średni udział składnika [%]")
    ax.set_title("A. Średnie udziały składników w rodzinach", pad=33)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=4, frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=.18)
    fig.tight_layout()
    save(fig, out, "srednie_udzialy_rodzin")


def pick(rows, family, **targets):
    candidates = [row for row in rows if row["family"] == family]

    def distance(row):
        return sum((np.log(row["params"][key] / value)) ** 2
                   for key, value in targets.items())

    return min(candidates, key=distance)


def family_examples(rows, metadata, out):
    for row in rows:
        row["params"] = metadata[row["curve_id"]]["params"]
    selected = [
        pick(rows, "F1_forced", rate=3),
        pick(rows, "F2_quench_forced", rate_before=30, rate_after=5, switch_C=500),
        next(row for row in rows if row["family"] == "F3_pairs"
             and row["params"]["hold_C"] == 575
             and row["params"]["hold_s"] == 100),
        pick(rows, "F4", rate_at_600=0.5, exponent=1.2),
    ]
    titles = [
        "F1: stała szybkość",
        "F2: zmiana szybkości",
        "F3: postój izotermiczny",
        "F4: stopniowe zwalnianie chłodzenia",
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), layout="constrained")
    for index, (axis, row, title) in enumerate(zip(axes.flat, selected, titles)):
        times, temperatures = curve_nodes(row["stages"])
        minutes = index == 3
        axis.plot(times / (60 if minutes else 1), temperatures,
                  linewidth=1.8, color="#2474A5")
        axis.set(title=title, xlabel=f"Czas [{'min' if minutes else 's'}]",
                 ylabel="Temperatura [°C]", ylim=(0, 860), xlim=(0, None))
        axis.set_yticks([20, 200, 400, 600, 830])
        axis.xaxis.set_major_locator(MaxNLocator(5))
        if index == 2:
            hold_times = times[1:3]
            axis.plot(hold_times, [575, 575], color="#DA7C30", linewidth=3)
            axis.annotate("100 s w 575°C", xy=(hold_times.mean(), 575),
                          xytext=(140, 720), fontsize=9,
                          arrowprops={"arrowstyle": "->", "color": "#555"})

    figure.savefig(out / "chlodzenie_rodziny.pdf", bbox_inches="tight")
    plt.close(figure)


def setup_axis(ax, end_minutes):
    ax.set(xlabel='Czas [min]', ylabel='Temperatura [°C]',
           xlim=(0, end_minutes * 1.03), ylim=(0, 850))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
    ax.grid(alpha=.2)


def f4_exponents(out):
    rate_at_600 = .5
    exponents = [1., 1.25, 1.5]
    settings = F4_SETTINGS
    curves = []
    colors = plt.get_cmap('viridis')(np.linspace(.12, .78, len(exponents)))
    for exponent, color in zip(exponents, colors):
        stages = smooth_stages(rate_at_600, exponent,
                               step=settings['temperature_step_C'], ambient=settings['ambient_C'])
        times, temperatures = curve_nodes(stages)
        curves.append(dict(p=exponent, color=color, stages=stages,
                           times=times, temperatures=temperatures))
    end_minutes = max(curve['times'][-1] for curve in curves) / 60
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    parameter_label = (f'$r_{{600}}$ = {rate_at_600:g} °C/s; '
                       f'$T_{{ot}}$ = {settings["ambient_C"]:g} °C')
    # The same absolute time axis preserves the differences in cooling duration.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    early_end = max(float(time_at_temperature(curve['stages'], 450.)) for curve in curves) / 60
    for ax in axes:
        for curve in curves:
            ax.plot(curve['times'] / 60, curve['temperatures'], color=curve['color'],
                    lw=1.7, label=f'$p$ = {curve["p"]:g}')
        setup_axis(ax, end_minutes)
    axes[0].set_title('Pełny przebieg chłodzenia')
    axes[0].legend(frameon=False)
    for curve in curves:
        axes[0].scatter(curve['times'][-1] / 60, T_END, color=curve['color'], s=20, zorder=3)
    axes[1].set(xlim=(0, early_end * 1.03), ylim=(430, 850), title='Początek chłodzenia')
    axes[1].axhline(600, color='.6', lw=.8, ls='--')
    axes[1].text(early_end * .98, 608, '600°C', ha='right', color='.4', fontsize=8)
    fig.suptitle(f'Wpływ wykładnika $p$ na krzywe F4\n{parameter_label}')
    fig.tight_layout()
    fig.savefig(out / "f4_porownanie_p.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, center, width, height, label, font_size=10):
    x, y = center
    rectangle = Rectangle(
        (x - width / 2, y - height / 2), width, height,
        facecolor='white', edgecolor='#222222', linewidth=1.1, zorder=2,
    )
    ax.add_patch(rectangle)
    ax.text(x, y, label, ha='center', va='center', fontsize=font_size,
            linespacing=1.3, color='#111111', zorder=3)


def draw_arrow(ax, points, label=None, label_position=None):
    # Explicit corners keep connections readable and clear of the boxes.
    codes = [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1)
    path = DrawingPath(points, codes)
    arrow = FancyArrowPatch(path=path, arrowstyle='->', mutation_scale=13,
                            linewidth=1.1, color='#333333', zorder=1)
    ax.add_patch(arrow)
    if label is not None:
        ax.text(*label_position, label, ha='center', va='center', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))


def draw_pipeline(rows, config, out):
    count = f'{len(rows):,}'.replace(',', ' ')
    fraction_limit = f"{config.reporting_quantum / 2:g}"
    sum_limit = f"{config.sum_tolerance:g}"

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.975)

    # First row: generation and preparation of simulator input.
    draw_box(ax, (20, 90), 22, 13, 'Generator krzywych')
    draw_box(ax, (50, 90), 23, 13, 'dataset.csv\nkrzywe bez etykiet')
    draw_box(ax, (80, 90), 23, 13, 'trans.dat\nplik wejściowy Trans_e')
    draw_arrow(ax, [(31, 90), (38.5, 90)])
    draw_arrow(ax, [(61.5, 90), (68.5, 90)])
    draw_arrow(ax, [(80, 83.5), (80, 77), (20, 77), (20, 70.5)])

    # Second row: the text report supplies eight fields; the file checks F/P/B/M.
    draw_box(ax, (20, 64), 22, 13, 'Trans_e.exe')
    draw_box(ax, (50, 64), 23, 15,
             'Raport tekstowy\n+ VolumeFractions.dat\n(kontrola F/P/B/M)', font_size=9.5)
    draw_box(ax, (82, 64), 30, 17,
             '8 pól raportu tekstowego:\nF, P, B, M, RA, cementyt,\nM odpuszczony,\nstopień przemiany', font_size=9.5)
    draw_arrow(ax, [(31, 64), (38.5, 64)])
    draw_arrow(ax, [(61.5, 64), (67, 64)])
    draw_arrow(ax, [(82, 55.5), (82, 49), (20, 49), (20, 43)])

    # Third row: qualification precedes normalization and assignment of labels.
    qualification = (
        'Kwalifikacja wyniku\n'
        f'RA i cementyt < {fraction_limit}\n'
        'pełna przemiana\n'
        f'suma F+P+B+M w granicach 1 ± {sum_limit}'
    )
    draw_box(ax, (20, 34), 38, 18, qualification, font_size=9.5)
    draw_box(ax, (55, 34), 23, 16,
             'Normalizacja F/P/B/M\nprzez ich sumę\n(np. 0.999 lub 1.001)', font_size=9.5)
    draw_box(ax, (84, 34), 25, 16,
             f'dataset.csv z etykietami\n{count} krzywych', font_size=9.5)
    draw_arrow(ax, [(39, 34), (43.5, 34)], 'tak', (41.25, 37))
    draw_arrow(ax, [(66.5, 34), (71.5, 34)])

    draw_box(ax, (20, 13), 34, 15,
             'Odrzucenie: wiersz zachowuje\nstatus i powód, bez etykiety\n(w tej kampanii nie wystąpiło)', font_size=9.5)
    draw_arrow(ax, [(20, 25), (20, 20.5)], 'nie', (23, 22.75))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'zbior_przeplyw.pdf', format='pdf', facecolor='white', metadata={
        'Title': 'Generowanie krzywych, symulacja Trans_e i kwalifikacja etykiet',
        'Subject': 'Przygotowanie zbioru krzywych chłodzenia',
    })
    plt.close(fig)


def plot_representations(row, cfg, out):
    stages = json.loads(row["stages_json"])
    times, temperatures = curve_nodes(stages)
    end = times[-1]
    k = 16
    # T(t) uses the same fixed time grid for every curve. Samples after the
    # cooling endpoint are still model inputs, with temperature equal to 20 C.
    time_samples = time_grid(k, cfg)
    display_end = max(end, time_samples[-1]) * 1.5
    query = np.unique(np.r_[np.geomspace(0.001, display_end, 4000), times, time_samples])
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), sharex=True, sharey=True)
    for ax, representation in zip(axes, ("tT", "Tt")):
        features, _ = prepare_features([stages], representation, k, cfg)
        if representation == "tT":
            sample_times = np.expm1(features[0])
            sample_temperatures = temperature_grid(k)
            title = "a) Czasy osiągania temperatur, t(T)"
        else:
            sample_times = time_samples
            sample_temperatures = features[0]
            title = "b) Temperatury w ustalonych chwilach, T(t)"
        ax.plot(query, temperature_at(stages, query), color="0.15", lw=1.8,
                label="Krzywa wejściowa symulatora")
        ax.plot(query, reconstruct_curve(features[0], representation, k, cfg, query),
                color="#21618c", lw=1.4, ls="--", label="Rekonstrukcja z K = 16 cech")
        ax.scatter(sample_times, sample_temperatures,
                   s=21, color="#b14c20", zorder=4, label="Próbki reprezentacji")
        ax.axvline(end, color="0.55", ls=":", lw=.9)
        ax.text(end, 850, " Koniec chłodzenia", fontsize=7, rotation=90,
                va="top", ha="right", color="0.4")
        ax.set_xscale("symlog", linthresh=.01)
        ax.set_xlim(0, display_end)
        ax.set_ylim(0, 870)
        ticks = [0, .01, .1, 1, 10, 100, 1000, 10000, 100000]
        ticks = [tick for tick in ticks if tick <= display_end]
        ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title, fontsize=10, loc="left", pad=10)
        ax.set_xlabel("Czas [s], skala logarytmiczna powyżej 0.01 s")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Temperatura [°C]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(out / 'reprezentacje_krzywej.pdf', bbox_inches='tight')
    plt.close(fig)


def draw(rows, metadata, config, out):
    for row in rows:
        row['stages'] = json.loads(row['stages_json'])
    with plt.rc_context():
        f4_exponents(out)
    with plt.rc_context():
        family_examples(rows, metadata, out)
    with plt.rc_context():
        draw_pipeline(rows, config, out)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'legend.fontsize': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'axes.axisbelow': True}):
        curve_overview(rows, metadata, out, seed=20261106, per_family=200)
        histograms(rows, out)
        family_phase_summary(rows, out)
    with plt.rc_context():
        example = next(row for row in rows if row['curve_id'] == '5898')
        plot_representations(example, config, out)
