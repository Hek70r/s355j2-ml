"""Figures 11–14 from the results and discussion chapters."""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

from src.config import PHASES, RESOLUTIONS


MODELS = ('rf', 'lightgbm', 'mlp', 'cnn1d', 'lstm', 'ridge')
NAMES = dict(rf='las losowy', lightgbm='LightGBM', mlp='MLP', cnn1d='CNN1D',
             lstm='LSTM', ridge='Ridge')
COLORS = dict(rf='#1575a9', lightgbm='#dd8a24', mlp='#32865e', cnn1d='#bd4c68',
              lstm='#7469c7', ridge='#595959')
REPS = {'Tt': r'$T(t)$', 'tT': r'$t(T)$'}


def decorate(ax):
    ax.grid(True, which='major', alpha=.2, linewidth=.6)
    ax.set_axisbelow(True)
    if ax.get_yscale() == 'linear':
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))


def decimal_ticks(ax):
    formatter = FuncFormatter(lambda value, position: f'{value:g}')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(NullFormatter())


def resolution_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set_xticks(RESOLUTIONS)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Liczba cech $K$')
    decorate(ax)


def mae_plot(variants, out):
    stats = variants[variants.model.isin(MODELS) &
                     variants['mode'].eq(np.where(variants.model == 'lightgbm', 'separate', 'joint'))]
    recommendations = stats[stats.selected_by_cv].set_index('model').to_dict('index')
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
    for ax, rep in zip(axes, REPS):
        for model in MODELS:
            points = stats[(stats.model == model) & (stats.representation == rep)].sort_values('k')
            ax.errorbar(points.k, points.MAE_mean, yerr=points.sd_pp.fillna(0),
                        color=COLORS[model], marker='o', markersize=3.5, capsize=2,
                        lw=1.2, label=NAMES[model])
            selected = recommendations[model]
            if selected['representation'] == rep:
                chosen = points[points.k == selected['k']].iloc[0]
                ax.scatter(chosen.k, chosen.MAE_mean, marker='*', s=100,
                           color=COLORS[model], edgecolors='black', linewidths=.4, zorder=5)
        ax.set(title=REPS[rep], yscale='log', ylabel='Testowe MAE [p.p.]')
        resolution_axis(ax)
        decimal_ticks(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.set_layout_engine('none')
    fig.savefig(out / 'mae_k.pdf', dpi=220, bbox_inches='tight')
    plt.close(fig)


def threshold_plot(plotted, variants, out):
    groups = [('Modele drzewiaste', ('rf', 'lightgbm')),
              ('Sieci neuronowe', ('cnn1d', 'mlp', 'lstm'))]
    markers = dict(rf='o', lightgbm='s', cnn1d='^', mlp='v', lstm='D')
    selected = variants[variants.selected_by_cv].set_index('model')
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex='col', sharey=True)
    for row, (group_name, group_models) in enumerate(groups):
        for col, (family, title, xlabel) in enumerate([
                ('F3_quench', 'F3: dojście hartownicze', 'Temperatura postoju [°C]'),
                ('F2_air_quench', 'F2: powietrze → hartownicze', 'Temperatura zmiany szybkości [°C]')]):
            ax = axes[row, col]
            for model in group_models:
                points = plotted[plotted.family.eq(family) & plotted.model.eq(model)]
                settings = selected.loc[model]
                rep = 'T(t)' if settings['representation'] == 'Tt' else 't(T)'
                model_label = 'random forest\n' if model == 'rf' else NAMES[model] + ' '
                label = f'{model_label}({rep}, K={settings["k"]})'
                ax.scatter(points.temperature_C, points.error_pp, color=COLORS[model], marker=markers[model],
                           s=18, alpha=.65, linewidths=0, label=label)
            ax.set_ylim(0, 10)
            ax.set_yticks(np.arange(0, 11, 2))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda value, position: f'{value:g}'))
            panel = plotted[plotted.family.eq(family) & plotted.model.isin(group_models)]
            above = int(panel.error_pp.gt(10).sum())
            if above:
                ax.text(.98, .96, f'Poza zakresem osi: {above}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            ax.set_title(f'{"abcd"[2 * row + col]}) {group_name}; {title}', fontsize=10)
            if row == 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel('Błąd krzywej [p.p.]')
                for value, label in [(406., '406°C'), (576.5, '576.5°C')]:
                    ax.axvline(value, color='#777777', ls='--', lw=.8, zorder=0)
                    ax.text(value + 5, .98, label, transform=ax.get_xaxis_transform(),
                            rotation=90, va='top', fontsize=8)
            ax.legend(loc='upper left', fontsize=8, framealpha=.92)
            ax.grid(True, which='major', alpha=.2, linewidth=.6)
            ax.set_axisbelow(True)
    fig.tight_layout(pad=1.2, h_pad=1.6)
    fig.savefig(out / 'strefy_temperatur_drzewa_sieci.pdf', dpi=220,
                bbox_inches='tight', pad_inches=.06)
    plt.close(fig)


def parity_plot(rf, out):
    colors = ('#3479ad', '#d98e25', '#45956a', '#b94a57')
    titles = ('Ferryt', 'Perlit', 'Bainit', 'Martenzyt')
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), layout='constrained')
    for ax, phase, title, color in zip(axes.flat, PHASES, titles, colors):
        ax.scatter(100 * rf[phase], 100 * rf[f'yhat_{phase}'],
                   s=8, alpha=.35, color=color, edgecolors='none', rasterized=True)
        ax.plot([0, 100], [0, 100], 'k--', lw=.8)
        ax.set(xlim=(-2, 102), ylim=(-2, 102), title=title,
               xlabel='Udział w etykiecie [%]', ylabel='Przewidywany udział [%]',
               xticks=[0, 25, 50, 75, 100], yticks=[0, 25, 50, 75, 100])
        ax.grid(alpha=.2, linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_aspect('equal')
    fig.savefig(out / 'zgodnosc_rf.pdf', dpi=200)
    plt.close(fig)


def hold_labels(dataset, metadata, out):
    frame = dataset[dataset.family.eq('F3_quench')].copy()
    frame['hold_C'] = [metadata[str(index)]['params']['hold_C'] for index in frame.index]
    frame['B_pct'], frame['M_pct'] = 100 * frame.B, 100 * frame.M
    windows = []
    for lower in (403, 406):
        group = frame[(frame.hold_C >= lower) & (frame.hold_C < lower + 3)]
        windows.append(dict(lower_C_inclusive=lower, upper_C_exclusive=lower + 3,
                            mean_B_pct=group.B_pct.mean(), mean_M_pct=group.M_pct.mean()))
    selected_windows = pd.DataFrame(windows)
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.45), sharey=True)
    for ax, title, limits in zip(axes, ('Cały blok F3: dojście hartownicze', 'Okolice 406°C'),
                                ((200, 800), (390, 422))):
        for phase, color, marker, label in [('B', '#45956a', 'o', 'Bainit'),
                                          ('M', '#b94a57', '^', 'Martenzyt')]:
            g = frame[frame.hold_C.between(*limits)]
            ax.scatter(g.hold_C, g[f'{phase}_pct'], s=10 if ax is axes[0] else 16,
                       alpha=.35 if ax is axes[0] else .65, color=color, marker=marker,
                       edgecolors='none', rasterized=True, label=label)
            if ax is axes[1]:
                for row in selected_windows.itertuples():
                    ax.hlines(getattr(row, f'mean_{phase}_pct'), row.lower_C_inclusive,
                              row.upper_C_exclusive, color=color, lw=2.7, zorder=4)
        ax.axvline(406., color='#555555', ls='--', lw=1)
        ax.set(xlim=limits, ylim=(-2, 102), title=title, xlabel='Temperatura postoju [°C]',
               ylabel='Udział składnika [%]')
        ax.legend(loc='upper right')
        ax.grid(alpha=.2, lw=.6)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'.replace('.', ',')))
    fig.tight_layout(pad=1.1)
    fig.savefig(out / 'etykiety_postoj.pdf', dpi=240, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)


def draw(dataset, metadata, results, out):
    variants = pd.read_csv(results / 'variants.csv')
    threshold = pd.read_csv(results / 'temperature_errors.csv')
    rf = pd.read_csv(results / 'rf_predictions.csv').join(dataset[list(PHASES)], on='curve_id')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        with plt.rc_context({'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8}):
            mae_plot(variants, out)
        threshold_plot(threshold, variants, out)
        with plt.rc_context({'axes.titlesize': 10, 'legend.fontsize': 8, 'pdf.fonttype': 42}):
            hold_labels(dataset, metadata, out)
    parity_plot(rf, out)
