# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [1.7.x]

## [1.7.2] - unreleased

### Changed

- The `main` branch is protected. (Github only, maybe a release is unnecessary?)

## [1.7.1] - 2025-06-03

### Added

- Created `release.py` script to automate the version releasing operations.
  - Usage: `python release.py <X.Y.Z>`

## [1.7.0] - 2025-05-15

### Added

- Support for the Retarding Field Analyzer
- `MultipactorTest.sweet_plot` method accepts a `masks` argument to show data with different linestyles.

### Changed

- Links to the `multipac_testbench` in the notebook tutorials should be clickable.

## [1.6.3] - 2025-04-23

### Added

- Examples in documentation.
- Package is available on PyPI and can be installed with `pip install multipac_testbench`.

## [1.6.2] - 2025-04-23

### Fixed

- Pre-commit hooks

## [1.6.1] - 2025-04-23

### Changed

- Better overall documentation.
- Documentation is now hosted on [ReadTheDocs](https://multipac-testbench.readthedocs.io/en/latest/)

### Fixed

- All the links in documentation are resolved.

## [1.6.0] - 2024-09-17

### Modified

- Proper packaging, local installation with pip.
 - You shall remove the `.src` in the `multipac_testbench` imports.

## [1.5.2] - 2024-03-06

### Added

- `TestCampaign.sweet_plot` now accepts the `all_on_same_plot: bool` kwarg. Associated example in Gallery.

### Modified

- `sweet_plot` and `plot_thresholds` now return the plotted Axes as well as the pd.DataFrame to produce it.
- Colors of instruments are set according to their pick-up.

### Fixed

- Last `MultipactorBand` was not added
- Sometimes, a `MultipactorBand` was incorrectly created because the previous multipactor starting index was not properly reinitialized

## [1.5.1] - 2024-03-02

### Removed

- `TestCampaign` methods:
  - `susceptiblity_chart`
  - `plot_instruments_vs_time`
  - `plot_instruments_y_vs_instrument_x`
  - `plot_data_at_multipactor_thresholds`
- `MultipactorTest` methods:
  - `plot_instruments_vs_time`
  - `plot_data_at_multipactor_thresholds`
  - `data_for_susceptibility`
  - `plot_instruments_y_vs_instrument_x`
- `Powers`


## [1.5.0] - 2024-03-02

### Added

- Instruments: `ForwardPower`, `ReflectedPower`, `SWR`, `ReflectionCoefficient`, `Frequency`
- `TestMultipactorBands`, `CampaignMultipactorBands` to handle when/where multipactor is detected in a more consistent way. Multipactor conditioned during test properly handled.
- `sweet_plot`, `plot_thresholds`, `at_last_threshold`, `susceptiblity` methods

### Modified

- `MultipactorBands` is now `InstrumentMultipactorBands`

### Deprecated

- `plot_instruments_vs_time`, `plot_instruments_y_vs_instrument_x` (use `sweet_plot` instead)
- `plot_data_at_multipactor_thresholds` (use `plot_thresholds` instead)
- `susceptiblity_chart` (use `susceptiblity` instead)
- `Powers` instrument

### Removed

 - For consistency, one `Instrument` = one column in the `MultipactorTest.df_data`.
 - Now, use the dedicated instruments: `ForwardPower`, `ReflectedPower` (both are `Power`), `SWR`, `ReflectionCoefficient`


## [1.4.1] - 2024-02-19

### Removed

- `TestCampaign.plot_multipactor_limits`
- `MultipactorTest.plot_multipactor_limits`
- `MultipactorTest._get_proper_multipactor_bands`
- `MultipactorTest.filter_measurement_points`
- In following methods, `multipactor_measured_at` is no longer supported. `MultipactorBands` object(s) must be given instead.
  - `MultipactorTest.data_for_somersalo`
  - `MultipactorTest.data_for_susceptibility`
- `Instruments.RfPower`
- `Instruments.SWR`


## [1.4.0] - 2024-02-14

### Added

- `MultipactorTest` has an `output_filepath` method for consistent output file naming.
- `MultipactorTest` has method `plot_data_at_multipactor_thresholds`, may replace `plot_multipactor_limits`
- New `TestCampaign` methods, calling their `MultipactorTest` counterpart recursively:
  - `reconstruct_voltage_along_line`
  - `animate_instruments_vs_position`
  - `plot_instruments_vs_time`
  - `scatter_instruments_data`
  - `plot_multipactor_limits`
  - `plot_data_at_multipactor_thresholds`
- You can now create `MultipactorBands` objects from several other `MultipactorBands` objects. Typical use cases:
 - At a pick-up with a `Penning` and a `CurrentProbe`, merge their multipactor bands.
 - Know when multipactor happens somewhere in the testbench, by merging all the detected multipactor bands.
- `TestCampaign.somersalo_scaling_law`, `TestCampaign.check_perez` to check some scaling laws

### Modified

- The `FieldProbe._patch_data` method modifies its `raw_data` instead of adding a `post_treater`, so that we can plot 'raw' field measurements that make any sense.
- `MultipactorTest.susceptiblity_plot` is `MultipactorTest.susceptiblity_chart` 
- `MultipactorTest.somersalo` is `MultipactorTest.somersalo_chart` 
- The `detect_multipactor` methods now return a list of `MultipactorBands` objects. Give it to `plot_multipactor_limits`, `plot_data_at_multipactor_thresholds`, `somersalo_chart`, `susceptiblity_chart` methods to explicitely link the plotted instruments to the multipacting bands.

### Deprecated

- The `multipactor_bands` attributes of `Instrument` and `IMeasurementPoint` will be removed. It will be mandatory to explicitely pass this argument when you want to plot multipactor limits.

### Fixed
- The fitting of the electric field over the probes now work correctly.

## [1.3.3] - 2024-02-10

### Added

- FieldProbe data can be reconstructed to avoid wrong G probe. Set `patch = True` and give a `calibration_file` in corresponding `.toml` entry.

## [1.3.2] - 2024-02-09

### Added

- `MultipactorTest` and `TestCampaign` accept `info` key to identify each test more easily.
- `MultipactorTest.get_instruments` handles more use cases.
- `MultipactorTest.plot_instruments_y_vs_instrument_x` method.

## [1.3.1] - 2024-02-04

### Added

- `power_is_growing` is now an attribute of `MultipactorBands`.

### Changed

- Calculation of when power is growing performed within `MultipactorTest.detect_multipactor()`. This methods accepts `power_is_growing_kw`.

### Removed

- `MultipactorTest.set_multipac_detector()`.

## [1.3.0] - 2024-02-03

### Added

- Position and timing of multipactor is now saved in `IMeasurementPoint.MultipactorBands` object, which is a list of `MultipactorBand` objects.
- `MultipactorTest.get_measurement_points()`
- `MultipactorTest.get_measurement_point()`
- A CHANGELOG.

### Changed

- `MultipactorTest.set_multipac_detector()` is now `MultipactorTest.detect_multipactor()`
- Only one multipactor instrument/criterion can be defined at the same time. Consequently, there is no need for precising the `multipactor_detector` keys in plotting funcs.

### Deprecated

- `MultipactorTest.filter_measurement_points()`, use `.get_measurement_points` instead.
- `MultipactorTest.set_multipac_detector()`, use `MultipactorTest.detect_multipactor()`
