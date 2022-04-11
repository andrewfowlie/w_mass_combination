"""
Discrepancy between combinatin and SM
=====================================
"""

from mw import weighted_least_squares, load


if __name__ == "__main__":

    d = load("mw.yml")
    c = weighted_least_squares(d)
    SM = d["sm"]

    Z = (c["mu"] - SM["mu"]) / (SM["sigma"]**2 + c["sigma"]**2)**0.5
    print("Discrepancy", Z)

    ZS = (c["mu"] - SM["mu"]) / (SM["sigma"]**2 + c["S"]**2 * c["sigma"]**2)**0.5
    print("Discrepancy after inflating error following PDG", ZS)
