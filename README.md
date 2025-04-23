# MULTIPAC testbench
This library is designed to post-treat the data from the MULTIPAC multipactor test bench at LPSC, Grenoble, France.

## Installation

### Users
1. Create a dedicated Python environment, activate it.
2. Run `pip install multipac_testbench`

### Developers
1. Clone the repository:
`git clone git@github.com:AdrienPlacais/multipac_testbench.git`
2. Create a dedicated Python environment, activate it.
3. Navigate to the main `multipac_testbench` folder and install the library with all dependencies: `pip install -e .`

Note that you will need Python 3.11 or higher to use the library.

If you want to use `conda`, you must manually install the required packages defined in `pyproject.toml`.
Then, add `multipac_testbench.src` to your `$PYTHONPATH` environment variable.

## Documentation

- Documentation is available on [ReadTheDocs](https://multipac-testbench.readthedocs.io/en/stable/).
- Examples are provided in the [Tutorials](https://multipac-testbench.readthedocs.io/en/stable/manual/tutorials.html) section.
  They all use the same `testbench_configuration.toml` and `120MHz-SWR4.csv` files that I can send upon request.

## Future updates

- [ ] Calibration of new field probes.
- [ ] Implementation of Retarding Field Analyzer.
