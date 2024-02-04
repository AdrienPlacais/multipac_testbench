# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.1] 2024.02.-1

### Added
- `power_is_growing` is now an attribute of `MultipactorBands`.

### Changed
- Calculation of when power is growing performed within `MultipactorTest.detect_multipactor()`. This methods accepts `power_is_growing_kw`.

### Removed

- `MultipactorTest.set_multipac_detector()`.


## [1.3.0] - 2024.02.03

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
