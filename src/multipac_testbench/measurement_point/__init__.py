"""Define objects to gather several :class:`.Instrument`.

In particular, :class:`.PickUp` gathers several :class:`.Instrument`
positioned at the same pick-up and :class:`.GlobalDiagnostics` gathers the
:class:`.Instrument` objects that are not positioned at a specific pick-up.
Both objects inherit from :class:`.IMeasurementPoint`, which defines all their
common properties and methods.

"""
