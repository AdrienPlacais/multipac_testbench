.. _presentation:

Introduction
************

.. toctree::
   :maxdepth: 2

This library offers simple methods to post-treat data from the MULTIPAC
testbench. Check :ref:`tutorials` for a quick tour of the available commands.
Go to `Gallery`_ for a quick overview of the available plots.

.. _Gallery: gallery.ipynb

Files
-----

In order to read the files produced by LabViewer, it is adviced to do the
following:

- Decimal should be changed from `,` to `.` (or use `decimal=','` when
  instantiating :class:`.MultipactorTest` or :class:`.TestCampaign`).
- Comments in the right-most columns should be deleted.

Note that you can use the column separator that you want with the `sep`
argument from the :class:`.MultipactorTest` or
:meth:`.TestCampaign.from_filepaths` methods.

.. note::
   Some text editors such as Windows Notepad are not adapted to big files such
   as the measurement files. Prefer Notepad++ or a spreadsheet editor.
