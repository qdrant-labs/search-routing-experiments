"""Hand-written realism overrides (SPEC d34b).

An override is a SurfaceGenerator subclass whose `feature` equals an
existing twin bank's emitted name; the registry shadows the pattern-sampled
default with it. Write one only where gibberish hurts (ticket prefixes,
plausible years) — which features need one is discovered empirically from
seeded round-trip samples (see SPEC deferred questions), never up front.
"""

from taxonomy_generators.core import SurfaceGenerator

OVERRIDES: tuple[type[SurfaceGenerator], ...] = ()
