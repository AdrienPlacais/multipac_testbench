.. _configuration:

Configuration
*************

.. toctree::
   :maxdepth: 2

Basics
------
The test bench setup is defined in a `TOML` file.
Here is a short example:

.. code-block:: toml

   # Define all the global instruments, that are not defined at a specific pick-up:
   [global]
   [global.instruments_kw]
   # You must link a file column name to an instrument, like this:
   [global.instruments_kw.NI9205_Power1]
   class_name = "ForwardPower"    # <- this instrument is mandatory

   [global.instruments_kw.NI9205_Power2]
   class_name = "ReflectedPower"  # <- second mandatory instrument

   # Define a pick-up; it needs a position
   [V1]
   position = 0.130

   # Define the instruments of the V1 pick-up
   [V1.instruments_kw]
   [V1.instruments_kw.NI9205_Penning1]
   class_name = "Penning"

   # You can have several instruments at the same pick-up:
   [E3]
   position = 0.39

   [E3.instruments_kw]
   [E3.instruments_kw.NI9205_MP3l]
   class_name = "CurrentProbe"

   [E3.instruments_kw.NI9205_E3]
   class_name = "FieldProbe"

You can check :data:`.STRING_TO_INSTRUMENT_CLASS` for the allowed names of instruments.

Raw files
---------
If you instantiate your objects with raw data, you must tell the code how to convert the acquisitions voltages to actual physical data.
Each instrument has a transfer function, which needs parameters defined in the `TOML`.
As an example, :class:`.FieldProbe` needs:
- `calibration_file` (see also: `MULTIPAC testbench calibrate racks`_)
- `attenuation_file`

.. _MULTIPAC testbench calibrate racks: https://github.com/AdrienPlacais/multipac_testbench_calibrate_racks


.. code-block:: toml

   [E1]
   position = 0e0

   [E1.instruments_kw]
   [E1.instruments_kw.NI9205_MP1l]
   class_name = "CurrentProbe"

   # Some instruments accept other arguments:
   [E1.instruments_kw.NI9205_E1]
   class_name = "FieldProbe"
   calibration_file = "../data/calibration/E1_fit_calibration.csv"
   attenuation_file = "../data/calibration/attenuations.csv"

See also: :ref:`load data <load>`.

Examples
--------
An example configurations is in the repository in `src/multipac_testbench/data/`.
