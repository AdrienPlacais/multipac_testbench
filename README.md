# MULTIPAC testbench
This library is designed to post-treat the data from the MULTIPAC multipactor test bench at LPSC, Grenoble, France.

## Installation

### Users
1. Create a dedicated Python environment, activate it.
2. Run `pip install multipac_testbench`

> [!NOTE]
> If you are completely new to Python and these instructions are unclear, check [this tutorial](https://python-guide.readthedocs.io/en/latest/).
> In particular, you will want to:
> 1. [Install Python](https://python-guide.readthedocs.io/en/latest/starting/installation/) 3.11 or higher.
> 2. [Learn to use Python environments](https://python-guide.readthedocs.io/en/latest/dev/virtualenvs/), `pipenv` or `virtualenv`.
> 3. [Install a Python IDE](https://python-guide.readthedocs.io/en/latest/dev/env/#ides) such as Spyder or VSCode.

### Developers
1. Clone the repository:
`git clone git@github.com:AdrienPlacais/multipac_testbench.git`
2. Create a dedicated Python environment, activate it.
3. Navigate to the main `multipac_testbench` folder and install the library with all dependencies: `pip install -e .`

Note that you will need Python 3.11 or higher to use the library.

If you want to use `conda`, you must manually install the required packages defined in `pyproject.toml`.
Then, add `multipac_testbench.src` to your `$PYTHONPATH` environment variable.

## Project info

- 📚 [Documentation](https://multipac-testbench.readthedocs.io/en/stable/)
- 📋 [Changelog](./CHANGELOG.md)
- 🤝 [Contributing](./CONTRIBUTING.md)
- 🚀 [Tutorials/examples](https://multipac-testbench.readthedocs.io/en/stable/manual/tutorials.html)

## Future updates

- [ ] Calibration of new field probes.
- [x] Implementation of Retarding Field Analyzer.
    - [x] RPA grid in V instead of kV.
- [ ] `sweet_plot` updates for better RPA treatment:
    - [ ] Allow for `head` argument, working similarly to `tail`.
    - [x] Argument to plot increasing and decreasing values differently. Useful when plotting RPA current against RPA grid voltage.
- [ ] Option to plot maximum of a signal per power cycle.
- [ ] Bugfix: Upper threshold is given even when we did not exit the multipactor zone.
