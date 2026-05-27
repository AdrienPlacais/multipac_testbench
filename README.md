# MULTIPAC testbench

This library is designed to post-treat the data from the MULTIPAC multipactor
test bench at LPSC, Grenoble, France.

## Installation

### Users

1. Create a dedicated Python environment, activate it.
2. Run `pip install multipac_testbench`

> [!NOTE]
> If you are completely new to Python and these instructions are unclear, check
> [this tutorial](https://python-guide.readthedocs.io/en/latest/). In
> particular, you will want to:
>
> 1. [Install
>    Python](https://python-guide.readthedocs.io/en/latest/starting/installation/)
>    3.12 or higher.
> 2. [Learn to use Python
>    environments](https://python-guide.readthedocs.io/en/latest/dev/virtualenvs/),
>    `pipenv` or `virtualenv`.
> 3. [Install a Python
>    IDE](https://python-guide.readthedocs.io/en/latest/dev/env/#ides) such as
>    Spyder or VSCode.

### Developers

1. Clone the repository:
   `git clone git@github.com:AdrienPlacais/multipac_testbench.git`
2. Create a dedicated Python environment, activate it.
3. Navigate to the main `multipac_testbench` folder and install the library
   with all dependencies: `pip install -e .`

Note that you will need Python 3.12 or higher to use the library.

If you want to use `conda`, you must manually install the required packages
defined in `pyproject.toml`. Then, add `multipac_testbench.src` to your
`$PYTHONPATH` environment variable.

## Project info

- 📚 [Documentation](https://multipac-testbench.readthedocs.io/en/stable/)
- 📋 [Changelog](./CHANGELOG.md)
- 🤝 [Contributing](./CONTRIBUTING.md)
- 🚀 [Tutorials/examples](https://multipac-testbench.readthedocs.io/en/stable/manual/tutorials.html)

## Future updates

- [x] Calibration of new field probes.
- [x] Implementation of Retarding Field Analyzer.
  - [x] RPA grid in V instead of kV.
- [ ] `sweet_plot` updates for better RPA treatment:
  - [ ] Allow for `head` argument, working similarly to `tail`.
  - [x] Argument to plot increasing and decreasing values differently. Useful
        when plotting RPA current against RPA grid voltage.
- [ ] Option to plot maximum of a signal per power cycle.
- [x] Bug fix: Upper threshold is given even when we did not exit the
      multipactor zone.
- [x] Adding a post-treater to `Power` instruments should be reflected in `SWR`
      and `ReflectionCoefficient` calculations.
- [x] Add notebook execution to normal test workflow.
  - [x] Add jupyter installation to `pip` deps for test
- [ ] Add filtering of the 50Hz noise (for `PowerStep` only)
- [ ] Interactive plots:
  - [ ] Make possible the visualization of `PowerStep` from a `MultipactorTest`,
        like in Labviewer.
  - [ ] Toggle raw/physical plot.
  - [ ] Post-treaters should add info to the plots: window, median, etc.

## Labviewer to-do

- [x] Automatic export binary -> xlsx or csv
- [x] Automatic export of individual power step files
- [x] Allow to take last value from individual power step file rather than
      highest
- [ ] Fix synchro of dBm column
- [x] Exported continuous files `CSV` are inconsistent with power step:
  - Continuous files:
    - `;` delim
    - `,` floating point separator
    - Named `RAW_MC_Data_YYYYMMDD_hhmmss.csv`
    - Column names are different from pulsed files:
      - `dBm` instead of `NI9205_dBm`
      - `MP1` instead of `NI9205_MP1l`
      - `MP1` instead of `NI9205_MP1l`
        - Speaking of that... Why the `l` at the end of the column name?
      - Header line starts with a comment character and a space: `#`
  - Power step folders:
    - Stored files have `\t` delimiter
    - Named `YYMMDD-hhmmss-blabla_RAW_CSV`
  - What I would like:
    - `.` floating point separator
    - `,` column delimiter
    - Name of file/folder always start by date and time in ISO 8601-ish format:
      - `2025-12-25T13-44-21`
      - Actual ISO 8601 format would be: `2025-12-25T13:44:21` but it would mess
        up Windows file naming conventions.
    - Complete file/folder name: `<ISOdate>_informations-typed-by-usr_RAW`
      (with `.csv` if it is continuous measurement)
    - Identical column names, no comment character at the start of the columns
      header.
