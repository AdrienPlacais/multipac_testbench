"""Execute documentation notebooks."""

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.notebook
@pytest.mark.parametrize(
    "notebook_file",
    [
        pytest.param("animate_multipactor_test.ipynb", marks=pytest.mark.slow),
        pytest.param("check_somersalo_scaling_law.ipynb"),
        pytest.param("compare_several_tests.ipynb"),
        pytest.param("load.ipynb"),
        pytest.param("plot_evolution_thresholds.ipynb"),
        # pytest.param("plot_rpa_data.ipynb"),
        pytest.param(
            "plot_signal_measured_by_instruments_vs_another_signal.ipynb"
        ),
        pytest.param("plot_signals_measured_by_instruments.ipynb"),
        pytest.param("plot_susceptibility.ipynb"),
        pytest.param("plot_voltage_thresholds.ipynb"),
        pytest.param("reconstruct_voltage.ipynb"),
    ],
)
def test_notebook_exec(notebook_file: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir("docs/manual/notebooks/")
    with open(notebook_file) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert (
                ep.preprocess(nb) is not None
            ), f"Got empty notebook for {notebook_file}"
        except Exception as e:
            assert False, f"Failed executing {notebook_file} with error:\n{e}"
