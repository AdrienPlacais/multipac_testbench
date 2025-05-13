#!/usr/bin/env python3
"""Define a classic workflow to study the RPA signals."""
import tomllib
from pathlib import Path

from multipac_testbench.instruments import RPA, RPACurrent, RPAPotential
from multipac_testbench.multipactor_test import MultipactorTest

if __name__ == "__main__":
    project = Path("../data/campaign_ERPA")
    config_path = Path(project, "testbench_configuration.toml")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    results_path = Path(project, "MVE5-120MHz-50Ohm-BDTcomp-ERPA1_0dBm.csv")
    multipactor_test = MultipactorTest(
        results_path, config, freq_mhz=120.0, swr=1.0, sep=","
    )

    power_is_growing_kw = {
        "minimum_number_of_points": 10,
        "n_trailing_points_to_check": 5,
    }

    # Plot RPA current vs RPA potential
    fig, axes = multipactor_test.sweet_plot(
        RPACurrent,
        xdata=RPAPotential,
        # power_is_growing_kw=power_is_growing_kw,
        tail=-16,
    )

    # Plot distribution
    # fig, axes = multipactor_test.sweet_plot(RPA, xdata=RPAPotential)
