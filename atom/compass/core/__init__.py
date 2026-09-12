"""Engine-agnostic modelling.

Nothing in this subpackage may import from ATOM. It deals in operator shapes and
durations, which makes it testable on a laptop and reusable outside a serving
run — a standalone analyser, or a comparison across hardware.
"""
