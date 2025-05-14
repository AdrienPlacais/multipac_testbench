#!/usr/bin/env python3
"""Define a classic workflow to study the RPA signals."""
import tomllib
from pathlib import Path

from multipac_testbench.instruments import (
    RPA,
    ForwardPower,
    RPACurrent,
    RPAPotential,
)
from multipac_testbench.multipactor_test import MultipactorTest

if __name__ == "__main__":
    project = Path("../data/campaign_ERPA")
    config_path = Path(project, "testbench_configuration.toml")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    results_path = Path(project, "MVE5-120MHz-50Ohm-BDTcomp-ERPA1_1dBm.csv")
    multipactor_test = MultipactorTest(
        results_path,
        config,
        freq_mhz=120.0,
        swr=1.0,
        sep=",",
        info="1dBm (175W)",
    )

    rpa_potential = multipactor_test.get_instrument(RPAPotential)
    assert isinstance(rpa_potential, RPAPotential)
    rpa_potential_growth_mask = rpa_potential.growth_mask(
        minimum_number_of_points=2, n_trailing_points_to_check=0
    )
    masks = {
        "__(potential grows)": rpa_potential_growth_mask,
        "__(potential decreases)": ~rpa_potential_growth_mask,
    }

    # Plot RPA current vs RPA potential
    fig, axes = multipactor_test.sweet_plot(
        RPACurrent, xdata=RPAPotential, masks=masks, drop_repeated_x=True
    )

    # Plot distribution
    rpa = multipactor_test.get_instrument(RPA)
    assert isinstance(rpa, RPA)
    rpa.data_as_pd.plot(
        x=0,
        y=1,
        grid=True,
        xlabel="Electrons energy [eV]",
        ylabel="Distribution",
        title=str(multipactor_test),
    )
