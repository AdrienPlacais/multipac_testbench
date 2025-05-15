.. _tutorials:

Tutorials
*********
When you want to study only one file, use the :py:class:`.MultipactorTest` object, as described in following tutorial.
When you want to study several files, prefer the :py:class:`.TestCampaign`, which is basically a list of :class:`.MultipactorTest`.
There is no tutorial for :py:class:`.TestCampaign`, but go check out the examples in the gallery.
Just know that every useful method from :py:class:`.MultipactorTest` has its equivalent in :py:class:`.TestCampaign` (same name, same arguments).
:py:class:`.TestCampaign` also has specific methods, such as :py:meth:`.susceptibility` or :py:meth:`.check_somersalo_scaling_law`.

.. toctree::
   :maxdepth: 2

   notebooks/plot_signals_measured_by_instruments.ipynb
   notebooks/plot_signal_measured_by_instruments_vs_another_signal.ipynb
   notebooks/plot_rpa_data.ipynb
   notebooks/compare_several_tests.ipynb

   notebooks/plot_evolution_thresholds.ipynb
   notebooks/plot_voltage_thresholds.ipynb

   notebooks/plot_susceptibility.ipynb
   notebooks/check_somersalo_scaling_law.ipynb

   notebooks/reconstruct_voltage.ipynb
   notebooks/animate_multipactor_test.ipynb

   notebooks/complete_study_example.ipynb
