"""
Combine and plot W boson mass data
==============================
"""

from distutils.spawn import find_executable
import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.stats import norm, chi2
from sorcery import dict_of


def combine(data):
    """
    @returns Data combined using weighted chi-squared
    """
    w = np.array([1. / d["sigma"]**2 for d in data.values() if d.get("combine", True)])
    x = np.array([d["mu"] for d in data.values() if d.get("combine", True)])

    mu = np.dot(w, x) / w.sum()
    sigma = w.sum()**-0.5

    df = len(x) - 1
    chi_squared = np.dot(w, (x - mu)**2)
    reduced = chi_squared / df

    p_value = chi2.sf(chi_squared, df=df)
    significance = norm.isf(p_value)

    return dict_of(mu, sigma, chi_squared, df, reduced, p_value, significance)

def plot(data):
    """
    Plots data for publication figure
    """

    # make space for labels

    _, ax = plt.subplots(1, 2,
                         sharey=True,
                         gridspec_kw={'width_ratios': [1, 2]},
                         figsize=(7, 7))
    plt.subplots_adjust(left=0.05, top=0.95, right=0.85, bottom=0.2, wspace=0., hspace=0.)

    for i, d in enumerate(reversed(data.values())):

        color = d.get("color", "black")

        # basic error bar

        ax[1].errorbar(d["mu"], i, xerr=d["sigma"], color=color, marker='s', capsize=3, ms=4)

        # vertical bar

        if d.get("emphasize"):
            x = [d["mu"] - d["sigma"], d["mu"] + d["sigma"]]
            ax[1].fill_between(x, y1=len(data) - 0.5, y2=-0.5, color=color, alpha=0.25, lw=0)
            ax[1].axvline(d["mu"], color=color, lw=1, ls="--", alpha=0.5)

        # labels with name and number

        left = d["label"]
        int_mu = int(round(d['mu']))
        int_sigma = int(round(d['sigma']))
        right = rf"${int_mu} \pm {int_sigma}$"

        if d.get("emphasize"):
            left = f"\\boldmath\\bfseries {left}"
            right = f"\\boldmath\\bfseries {right}"

        center = i - 0.075
        ax[0].text(0, center, left, color=color, horizontalalignment="left", fontsize=14)
        ax[1].text(80590, center, right, color=color, horizontalalignment="left", fontsize=14)

    # adjust visible axes
    ax[0].set_axis_off()
    ax[1].set_facecolor("none")
    ax[1].axes.get_yaxis().set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)

    # axis limits
    ax[1].set_ylim(-0.5, len(data) - 0.5)
    xlim = (80300, 80600)
    ax[1].set_xlim(xlim)

    # axis ticks
    xticks = [xlim[0], 80400, 80500, xlim[1]]
    ax[1].set_xticks(xticks)
    xticklabels = [str(x) for x in xticks]
    xticklabels[0] = xticklabels[-1] = ""
    ax[1].set_xticklabels(xticklabels)

    # axis labels
    ax[1].set_xlabel("$M_W$ [MeV]")

    # footnote
    footnote = "${}^*$ Does not include $13.5\,\textrm{MeV}$ shift in CDF 2002-2007 (2.2/fb)"
    ax[0].text(0, -3., footnote, horizontalalignment="left", fontsize=10)

if __name__ == "__main__":

    # combination

    with open("mw.yml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    combination = combine(data)
    print(combination)

    # plot

    data["our_combination"].update(combination)

    if find_executable('pdflatex'):
        plt.rcParams.update({'font.family': 'serif',
                             'text.usetex': True,
                             'font.size': 18})
    plot(data)
    plt.savefig("mw.pdf")
