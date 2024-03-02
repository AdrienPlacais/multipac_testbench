.. _presentation:

Introduction
************

.. toctree::
   :maxdepth: 2

This library offers simple methods to post-treat data from the MULTIPAC testbench.
Check :ref:`tutorials` for a quick tour of the available commands.
Go to `Gallery`_ for a quick overview of the available plots.

.. _Gallery: gallery.ipynb

Files
-----
A few operations are mandatory in order to read the files produced by LabVIEWER.
 - Mandatory: decimal should be changed from `,` to `.`.
 - Recommended: measurement index should start at `i=0`.
 - Recommended: comments in the right-most columns should be deleted.

Note that you can use the column separator that you want with the `sep` argument from the :py:meth:`.MultipactorTest.__init__` or :py:meth:`.TestCampaign.from_filepaths` methods.

.. note::
   Some text editors such as Windows Notepad are not adapted to big files such as the measurement files.
   Prefer Notepad++ or a spreadsheet editor.

.. todo::
   LabVIEWER data starts at index == 1! Fix this, every index list should always start at 0.
