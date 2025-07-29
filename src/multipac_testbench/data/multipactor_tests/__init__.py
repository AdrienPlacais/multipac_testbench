"""Provide data for testing purposes.

These files can be generated:
- With the :meth:`.PowerstepSet.to_multipactor_test_file`
- Manually from LabViewer.

"""

from importlib import resources

dir = resources.files(__name__)
test_140MHz_SWR4_11 = dir / "2025.06.20_140MHz-SWR4-11.csv"
test_140MHz_SWR3_12 = dir / "2025.06.20_140MHz-SWR3-12.csv"
test_140MHz_SWR2_13 = dir / "2025.06.20_140MHz-SWR2-13.csv"
test_140MHz_SWR1_14 = dir / "2025.06.20_140MHz-SWR1-14.csv"
test_120MHz_SWR1_4 = dir / "2025.06.19_120MHz-SWR1-4.csv"
