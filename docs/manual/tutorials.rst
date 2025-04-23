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

   notebooks/basics.ipynb
