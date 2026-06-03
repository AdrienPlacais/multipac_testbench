# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-06-03

### Added

- New `StepConstant`: `PreTrigger`, `PostTrigger`, `Trigger`, `CurrentCalibre`.
- New `Instrument`: `Sync`, holding the trigger command. Has a `trigger`
  property that computes the value of the trigger.

### Changed

- `HeaderConstant` is now `StepConstant`
- Frequency should be given in `TOML` via a `FrequencySetpoint`, even though
  it can still be given as a `MultipactorTest` argument.
- Old `PowerSetpoint` inheriting from `VirtualInstrument` are removed. They are
  replaced by equivalent `PowerSetpoint` inheriting from `StepConstant`.

### Removed

- `Frequency` implementation. This is now an alias to `FrequencySetpoint`.

## [2.0.0] - 2026-06-03

### Added

- `HeaderConstant` instruments read data from power step individual `CSV` files.
  - Currently, they are three to be implemented: `PolarizationSetpoint`,
    `NewPowerSetpoint` (will eventually replace `PowerSetpoint`),
    `FrequencySetpoint`.

```toml

#                      Instrument name. Will be name of column in exported
#                      `MultipactorTest` `CSV` file.
[global.instruments_kw.SM300_Level]
class_name = "NewPowerSetpoint"  # Class name
header_key = "SM300_Level"       # Value to read from power step `CSV` header

[global.instruments_kw.SM300_Frequency]
class_name = "FrequencySetpoint"
header_key = "SM300_Frequency"

#                      Note that for this one, we decided to change the name
#                      in the `CSV` header
[global.instruments_kw.NI9205_Polarization]
class_name = "PolarizationSetpoint"
header_key = "Polarisation_2"
```

> [!WARNING]
> Very soon, frequency will be read from the instrument instead of the user
> argument to `MultipactorTest`. It may break compatibility of older scripts,
> as they do not define the frequency setpoint instrument in their `TOML`.

### Fixed

- An `Instrument` declared in the `TOML` but missing from the data files do not
  make the code crash anymore.

## [1.10.2] - 2026-06-01

### Fixed

- `kwarg` name mismatch in `PowerStepSet`: use `are_raw`, not `is_raw`.

## [1.10.1] - 2026-05-28

### Added

- Vertical line in interactive plots to show x-position of the mouse.
- Possibility to switch between raw data and physical data in interactive plots.

### Fixed

- `MultipactorTest` created by `PowerStepSet.to_multipactor_test` where always
  raw.

## [1.10.0] - 2026-05-27

### Added

- `PowerStep.sweet_plot` accepts arguments `pre_trig` and `trig`, indicating
  first index of power pulse and power pulse duration. If these arguments are
  provided, position of the pulse is plotted.
- `PowerStepSet.to_multipactor_test` to create a `MultipactorTest` from a
  `PowerStepSet`.
- `MultipactorTest.interactive_sweet_plot` that allows to click the plot, a new
  figure opens with the associated `PowerStep`. Only works if `MultipactorTest`
  was created using `PowerStepSet.to_multipactor_test`.

## [1.9.1] - 2026-03-11

### Changed

- Modernized gh actions. PyPI release should go smoothly.

### Fixed

- Documentation builds.

## [1.9.0] - 2026-03-10

### Added

- `DiffPenning` `VirtualInstrument` that holds pressure variations.
- `quantity_is_above_local_average` multipactor detector.
  - Use `scripts.multipactor_detection_quantity` to tune it.

### Changed

- Filtering of instruments is now performed thanks to predicates passed to
  `MultipactorTest.get_instruments`:

  ```python
  # The predicates are defined in instruments.predicates
  predicate = instrument_name_selector(("NI9205_MP5l", "NI9205_MP6l"))
  some_current_probes = multipactor_test.get_instruments(predicate=predicate)

  predicate = instrument_type_selector(CurrentProbe)
  all_current_probes = multipactor_test.get_instruments(predicate=predicate)

  # Note that several class/names can be given to these predicates
  predicate = instrument_type_selector([CurrentProbe, FieldProbe])
  # Other examples
  predicate = measurement_point_excluder(["PU4", "PU8"])
  ```

  - The actual function called under the hood is `filter` in the
    `instruments.predicates` module.

- Use `instruments.predicates.combine_predicates` function to combine several
  predicates:

  ```python
  predicate_current = instrument_type_selector(CurrentProbe)
  predicate_exclud = instrument_excluder("NI9205_MP5l")
  # Will return all current probes but the one named 'NI9205_MP5l'
  predicate = combine_predicates(predicate_current, predicate_exclud)

  ```

### Deprecated

- The `measurement_points_to_exclude`, `instruments_to_ignore` keyword
  arguments Prefer passing predicates created with `instrument_excluder`,
  `measurement_point_excluder`. Methods that used these keywords:
  - `MultipactorTest.get_instruments()`
  - `MultipactorTest.determine_thresholds()`
  - `MultipactorTest.sweet_plot()` (`exclude` keyword)
  - `TestCampaign.determine_thresholds()`

### Fixed

- `dBm` of `PowerStep` is read from the header instead of the file name.
- `sweet_plot` methods plotted `ThresholdSet` multiple times when `ydata`
  designated several instruments.

## [1.8.2] - 2025-08-01

### Fixed

- Missing dependency when running `pytest` from GitHub actions.

## [1.8.1] - 2025-08-01

### Fixed

- Updated paths to notebooks used in unit testing.

## [1.8.0] - 2025-08-01

### Added

- `PowerStep` and `PowerStepSet` to handle power step (trigger) files.
  - Allows averaging instrument data, rather than taking the maximum!
- You can load `RAW` files.
  - Allows to control the `V_acquisition`➡️ `physical quantity` transfer function.
  - When loading `PowerStep` (trigger) files, it is more robust to performed
    averaging on measured current rather than on the physical quantity.
- New `Instrument` s:
  - `PowerSetpoint` to hold the `NI9205_dBm` column. Allows for a much more
    robust detection of power cycles and power extrema.
  - `FieldPowerError` to compute error between field measured by probes, and
    field calculated from powers.
- `ThresholdSet` object, containing all `Threshold` of a `MultipactorTest`.
- `AveragedThresholdTest`, derived from `ThresholdSet` to get median of several
  `Threshold`.
- Function to fix the voltage columns when rf rack calibration or probe
  attenuation was not updated in LabView.
  - Loading `RAW` files and using built-in transfer functions is however more
    robust.

### Changed

- When possible, we determine whether power is growing or not using
  `PowerSetpoint` rather than `ForwardPower`.
  - This is much more robust, as `NI9205_dBm` column is less noisy
    `NI9205_Power1`.

### Removed

- `MultipactorBand` objects removed, and replaced by `Threshold` and
  `ThresholdSet`. This is much more robust, as `MultipactorBand` implied the
  existence of two consecutive `Threshold` objects.

## [1.7.4] - 2025-06-23

### Added

- `trigger_policy` keyword at creation of `TestCampaign` or `MultipactorTest`,
  to handle several contiguous same power measurement points.

## [1.7.3] - 2025-06-16

### Added

- New post-treater to set data to a constant value where under a threshold.
- Updating `ForwardPower` or `ReflectedPower` automatically updates `SWR` and
  `ReflectionCoefficient`.
  - Used `Observer` design pattern.

## [1.7.2] - 2025-06-03

### Added

- `release.py` handles post-release steps related to `main` branch.

## [1.7.1] - 2025-06-03

### Added

- Created `release.py` script to automate the version releasing operations.
  - Usage: `python release.py <X.Y.Z>`

## [1.7.0] - 2025-05-15

### Added

- Support for the Retarding Field Analyzer
- `MultipactorTest.sweet_plot` method accepts a `masks` argument to show data
  with different linestyles.

### Changed

- Links to the `multipac_testbench` in the notebook tutorials should be clickable.

## [1.6.3] - 2025-04-23

### Added

- Examples in documentation.
- Package is available on PyPI and can be installed with `pip install
multipac_testbench`.

## [1.6.2] - 2025-04-23

### Fixed

- Pre-commit hooks

## [1.6.1] - 2025-04-23

### Changed

- Better overall documentation.
- Documentation is now hosted on
  [ReadTheDocs](https://multipac-testbench.readthedocs.io/en/latest/)

### Fixed

- All the links in documentation are resolved.

## [1.6.0] - 2024-09-17

### Modified

- Proper packaging, local installation with pip.
- You shall remove the `.src` in the `multipac_testbench` imports.

## [1.5.2] - 2024-03-06

### Added

- `TestCampaign.sweet_plot` now accepts the `all_on_same_plot: bool` kwarg.
  Associated example in Gallery.

### Modified

- `sweet_plot` and `plot_thresholds` now return the plotted Axes as well as the
  pd.DataFrame to produce it.
- Colors of instruments are set according to their pick-up.

### Fixed

- Last `MultipactorBand` was not added
- Sometimes, a `MultipactorBand` was incorrectly created because the previous
  multipactor starting index was not properly reinitialized

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

- Instruments: `ForwardPower`, `ReflectedPower`, `SWR`,
  `ReflectionCoefficient`, `Frequency`
- `TestMultipactorBands`, `CampaignMultipactorBands` to handle when/where
  multipactor is detected in a more consistent way. Multipactor conditioned
  during test properly handled.
- `sweet_plot`, `plot_thresholds`, `at_last_threshold`, `susceptiblity` methods

### Modified

- `MultipactorBands` is now `InstrumentMultipactorBands`

### Deprecated

- `plot_instruments_vs_time`, `plot_instruments_y_vs_instrument_x` (use
  `sweet_plot` instead)
- `plot_data_at_multipactor_thresholds` (use `plot_thresholds` instead)
- `susceptiblity_chart` (use `susceptiblity` instead)
- `Powers` instrument

### Removed

- For consistency, one `Instrument` = one column in the
  `MultipactorTest.df_data`.
- Now, use the dedicated instruments: `ForwardPower`, `ReflectedPower` (both
  are `Power`), `SWR`, `ReflectionCoefficient`

## [1.4.1] - 2024-02-19

### Removed

- `TestCampaign.plot_multipactor_limits`
- `MultipactorTest.plot_multipactor_limits`
- `MultipactorTest._get_proper_multipactor_bands`
- `MultipactorTest.filter_measurement_points`
- In following methods, `multipactor_measured_at` is no longer supported.
  `MultipactorBands` object(s) must be given instead.
  - `MultipactorTest.data_for_somersalo`
  - `MultipactorTest.data_for_susceptibility`
- `Instruments.RfPower`
- `Instruments.SWR`

## [1.4.0] - 2024-02-14

### Added

- `MultipactorTest` has an `output_filepath` method for consistent output file
  naming.
- `MultipactorTest` has method `plot_data_at_multipactor_thresholds`, may
  replace `plot_multipactor_limits`
- New `TestCampaign` methods, calling their `MultipactorTest` counterpart
  recursively:
  - `reconstruct_voltage_along_line`
  - `animate_instruments_vs_position`
  - `plot_instruments_vs_time`
  - `scatter_instruments_data`
  - `plot_multipactor_limits`
  - `plot_data_at_multipactor_thresholds`
- You can now create `MultipactorBands` objects from several other
  `MultipactorBands` objects. Typical use cases:
- At a pick-up with a `Penning` and a `CurrentProbe`, merge their multipactor
  bands.
- Know when multipactor happens somewhere in the testbench, by merging all the
  detected multipactor bands.
- `TestCampaign.somersalo_scaling_law`, `TestCampaign.check_perez` to check
  some scaling laws

### Modified

- The `FieldProbe._patch_data` method modifies its `raw_data` instead of adding
  a `post_treater`, so that we can plot 'raw' field measurements that make any
  sense.
- `MultipactorTest.susceptiblity_plot` is `MultipactorTest.susceptiblity_chart`
- `MultipactorTest.somersalo` is `MultipactorTest.somersalo_chart`
- The `detect_multipactor` methods now return a list of `MultipactorBands`
  objects. Give it to `plot_multipactor_limits`,
  `plot_data_at_multipactor_thresholds`, `somersalo_chart`, `susceptiblity_chart`
  methods to explicitely link the plotted instruments to the multipacting bands.

### Deprecated

- The `multipactor_bands` attributes of `Instrument` and `IMeasurementPoint`
  will be removed. It will be mandatory to explicitely pass this argument when
  you want to plot multipactor limits.

### Fixed

- The fitting of the electric field over the probes now work correctly.

## [1.3.3] - 2024-02-10

### Added

- FieldProbe data can be reconstructed to avoid wrong G probe. Set `patch =
True` and give a `calibration_file` in corresponding `.toml` entry.

## [1.3.2] - 2024-02-09

### Added

- `MultipactorTest` and `TestCampaign` accept `info` key to identify each test
  more easily.
- `MultipactorTest.get_instruments` handles more use cases.
- `MultipactorTest.plot_instruments_y_vs_instrument_x` method.

## [1.3.1] - 2024-02-04

### Added

- `power_is_growing` is now an attribute of `MultipactorBands`.

### Changed

- Calculation of when power is growing performed within
  `MultipactorTest.detect_multipactor()`. This methods accepts
  `power_is_growing_kw`.

### Removed

- `MultipactorTest.set_multipac_detector()`.

## [1.3.0] - 2024-02-03

### Added

- Position and timing of multipactor is now saved in
  `IMeasurementPoint.MultipactorBands` object, which is a list of
  `MultipactorBand` objects.
- `MultipactorTest.get_measurement_points()`
- `MultipactorTest.get_measurement_point()`
- A CHANGELOG.

### Changed

- `MultipactorTest.set_multipac_detector()` is now
  `MultipactorTest.detect_multipactor()`
- Only one multipactor instrument/criterion can be defined at the same time.
  Consequently, there is no need for precising the `multipactor_detector` keys in
  plotting funcs.

### Deprecated

- `MultipactorTest.filter_measurement_points()`, use `.get_measurement_points`
  instead.
- `MultipactorTest.set_multipac_detector()`, use
  `MultipactorTest.detect_multipactor()`
