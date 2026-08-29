"""The ATOM-aware half of Compass.

Everything that knows about ``ModelRunner``, ``ScheduledBatch`` or ATOM's
configuration lives here. ``atom.compass.core`` stays free of those imports so
it can be tested and reused on its own.
"""
