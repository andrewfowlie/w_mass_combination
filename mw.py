"""
Combine and plot W boson mass data
==================================
"""

from distutils.spawn import find_executable
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from scipy.stats import norm, chi2
from sorcery import dict_of


def weighted_least_squares(data):
    """
    See Sec. 5.2 of https://pdg.lbl.gov/2021/reviews/rpp2021-rev-rpp-intro.pdf

    @returns Data combined using weighted least-squares
    """
    w = np.array([1. / d["sigma"]**2 for d in data.values() if d.get("combine", True)])
    x = np.array([d["mu"] for d in data.values() if d.get("combine", True)])

    mu = np.dot(w, x) / w.sum()
    sigma = w.sum()**-0.5

    df = len(x) - 1
    chi_squared = np.dot(w, (x - mu)**2)
    reduced = chi_squared / df

    if reduced > 1.:
        delta0 = 3. * len(x)**0.5 * sigma
        where = w**-0.5 < delta0
        reduced_df = sum(where) - 1
        S = (np.dot(w[where], (x[where] - mu)**2) / reduced_df)**0.5
    else:
        S = 1.

    p_value = chi2.sf(chi_squared, df=df)
    significance = norm.isf(p_value)

    return dict_of(mu, sigma, chi_squared, df, reduced, p_value, significance, S)

def plot(data):
    """
    Plots data for publication figure
    """

    # make space for labels

    _, ax = plt.subplots(1, 2,
                         sharey=True,
                         gridspec_kw={"width_ratios": [1, 2]},
                         figsize=(7, 7))
    plt.subplots_adjust(left=0.05, top=0.95, right=0.85, bottom=0.2, wspace=0., hspace=0.)

    plot_data = [d for d in data.values() if d.get("plot", True)]

    for i, d in enumerate(reversed(plot_data)):

        color = d.get("color", "black")

        # basic error bar

        ax[1].errorbar(d["mu"], i, xerr=d["sigma"], color=color, marker='s', capsize=3, ms=4)

        # vertical bar

        if d.get("emphasize"):
            x = [d["mu"] - d["sigma"], d["mu"] + d["sigma"]]
            ax[1].fill_between(x, y1=len(plot_data) - 0.5, y2=-0.5, color=color, alpha=0.25, lw=0)
            ax[1].axvline(d["mu"], color=color, lw=1, ls="--", alpha=0.5)

        # labels with name and number

        left = d["label"]
        int_mu = int(round(d["mu"]))
        int_sigma = int(round(d["sigma"]))
        right = rf"${int_mu} \pm {int_sigma}$"

        if d.get("emphasize") and plt.rcParams['text.usetex']:
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
    ax[1].set_ylim(-0.5, len(plot_data) - 0.5)
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
    footnote = "${}^*$ Does not include 13.5 MeV shift in CDF 2002-2007 (2.2/fb)"
    ax[0].text(0, -3., footnote, horizontalalignment="left", fontsize=10)

def load(file_name):
    """
    @reurns Data from yaml file
    """
    with open(file_name) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    for d in data.values():
        try:
            d["sigma"] = np.linalg.norm(d["sigma"])
        except KeyError:
            pass
    return data

if __name__ == "__main__":

    # combination

    try:
        file_name = sys.argv[1]
    except IndexError:
        file_name = "mw.yml"

    data = load(file_name)
    combination = weighted_least_squares(data)
    print(combination)

    # plot

    data["our_combination"].update(combination)

    if find_executable("pdflatex"):
        plt.rcParams.update({"font.family": "serif",
                             "text.usetex": True,
                             "font.size": 18})
    plot(data)
    plt.savefig("mw.pdf")
