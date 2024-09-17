#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tomllib
from functools import partial
from pathlib import Path

from multipac_testbench.src.test import Test

# project = Path(
    # "/home/placais/Documents/Projects/Multipac/2024.01.23_EPISAMA_campaign_with_Yanis/data")
# config_path = Path(project, "newtestbench_configuration.toml")

project = Path(
    "/home/placais/Documents/Simulation/python/multipac_testbench/docs/source/manual/data/"
        )
filepath = project / "120MHz-SWR4.csv"

project = Path("/home/placais/Documents/Simulation/python/multipac_testbench/src/experimental/")
config_path = Path(project, "example_configuration.toml")
with open(config_path, "rb") as f:
    config = tomllib.load(f)

freq_mhz = 120.
swr = 4.

test = Test(filepath, config, freq_mhz, swr, sep=',')
data = test.data
metadata = test.metadata

# test.filter('instrument_type',
#             ('ElectricFieldProbe', 'Penning')).plot()
# test.filter('position', range(0., 1.)).plot()

toplot = ('voltage', 'pressure', 'power')
axes = test.plot_instruments_vs_time(toplot)

grouper = metadata.T.groupby('nature')
grouped = grouper.get_group('voltage')
index = grouped.index
data[index].plot()
