.. _configuration:

Configuration
*************

.. toctree::
   :maxdepth: 2

The test bench setup is defined in a ``.toml`` file.
Here is a short example:

.. code-block:: toml

   [global]
   [global.instruments_kw]
   [global.instruments_kw.NI9205_Power1]
   class_name = "ForwardPower"

   [global.instruments_kw.NI9205_Power2]
   class_name = "ReflectedPower"

   [E3]
   position = 0.39

   [E3.instruments_kw]
   [E3.instruments_kw.NI9205_MP3l]
   class_name = "CurrentProbe"

   [E3.instruments_kw.NI9205_E3]
   class_name = "FieldProbe"


You can check :data:`.STRING_TO_INSTRUMENT_CLASS` for the allowed names of instruments.

.. todo::
   Proper documentation for the configuration.
