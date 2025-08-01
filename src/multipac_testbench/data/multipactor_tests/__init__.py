"""Provide data for testing purposes.

These files can be generated:
- With the :meth:`.PowerStepSet.to_multipactor_test_file`
- Manually from LabViewer.

"""

from importlib import resources

dir = resources.files(__name__)
test_120MHz_SWR1_4 = dir / "2025.06.19_120MHz-SWR1-4.csv"
test_120MHz_SWR5_6 = dir / "2025.06.20_120MHz-SWR5-6.csv"
test_120MHz_SWR4_7 = dir / "2025.06.20_120MHz-SWR4-7.csv"
test_120MHz_SWR3_8 = dir / "2025.06.20_120MHz-SWR3-8.csv"
test_120MHz_SWR2_9 = dir / "2025.06.20_120MHz-SWR2-9.csv"
test_120MHz_SWR1_10 = dir / "2025.06.20_120MHz-SWR1-10.csv"
tests_120 = (
    test_120MHz_SWR5_6,
    test_120MHz_SWR4_7,
    test_120MHz_SWR3_8,
    test_120MHz_SWR2_9,
    test_120MHz_SWR1_10,
)

test_140MHz_SWR4_11 = dir / "2025.06.20_140MHz-SWR4-11.csv"
test_140MHz_SWR3_12 = dir / "2025.06.20_140MHz-SWR3-12.csv"
test_140MHz_SWR2_13 = dir / "2025.06.20_140MHz-SWR2-13.csv"
test_140MHz_SWR1_14 = dir / "2025.06.20_140MHz-SWR1-14.csv"
tests_140 = (
    test_140MHz_SWR4_11,
    test_140MHz_SWR3_12,
    test_140MHz_SWR2_13,
    test_140MHz_SWR1_14,
)

test_160MHz_SWR1_15 = dir / "2025.06.20_160MHz-SWR1-15.csv"
test_160MHz_SWR2_16 = dir / "2025.06.20_160MHz-SWR2-16.csv"
test_160MHz_SWR3_17 = dir / "2025.06.20_160MHz-SWR3-17.csv"
test_160MHz_SWR4_18 = dir / "2025.06.20_160MHz-SWR4-18.csv"
tests_160 = (
    test_160MHz_SWR1_15,
    test_160MHz_SWR2_16,
    test_160MHz_SWR3_17,
    test_160MHz_SWR4_18,
)
tests = tests_120 + tests_140 + tests_160
