# Installation
`git clone` the library to a folder in your `$PYTHONPATH`.
Alternatively, manually download and unzip it in a folder accessible by your `$PYTHONPATH`.
The name of the library must be `multipac_testbench`.

# Requirements
This library requires a modern version of Python installed (at least Python 3.9).
Some non-standard libraries are required, namely:
 - `matplotlib`
 - `numpy`
 - `pandas`
 - `scipy`

If you version of Python is 3.9 or 3.10, the `toml` parser `tomllib` will not be available.
You can use any `toml` parser, such as `tomli`.
In this case, replace every occurrence of `tomllib` by `tomli`.
