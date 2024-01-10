# Installation
`git clone` the library to a folder in your `$PYTHONPATH`.
Alternatively, manually download and unzip it in a folder accessible by your `$PYTHONPATH`.
The name of the library must be `multipac_testbench`.

# Requirements
This library requires a modern version of Python installed (at least Python 3.11).
Some non-standard libraries are required, namely:
 - `matplotlib`
 - `numpy`
 - `pandas`
 - `scipy`

# If Python 3.11 is not an option
The `toml` parser `tomllib` will not be available.
You can use any `toml` parser, such as `tomli`.
In this case, replace every occurrence of `tomllib` by `tomli`.

Some type hints will not be supported.
In all files, you should replace:
 - `from typing import Self` to remove
 - `-> Self:` to `:`
(maybe other type hints will not be supported, delete it as well)
