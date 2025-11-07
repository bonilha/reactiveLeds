# effects/__init__.py
from .basics import (
    effect_line_spectrum,
    effect_mirror_spectrum,
    effect_rainbow_wave,
    effect_energy_comets,
)
from .dynamics import (
    effect_peak_hold_columns,
    effect_full_strip_pulse,
    effect_waterfall,
    effect_bass_ripple_pulse_v2,
)
from .clean import (
    effect_spectral_blade,
    effect_bass_center_bloom,
    effect_peak_dots,
    effect_centroid_comet,
    effect_beat_outward_burst,
    effect_quantized_sections,
)
from .fire import (
    effect_clean_fire_edge_v3,
    effect_clean_fire_center_v3,
)

def _wrap(ctx, fn):
    return lambda bands, beat, active, _f=fn: _f(ctx, bands, beat, active)

def build_effects(ctx):
    return [
        ("Line Spectrum (HSV)",         _wrap(ctx, effect_line_spectrum)),
        ("Mirror Spectrum (HSV)",       _wrap(ctx, effect_mirror_spectrum)),
        ("Rainbow Wave (HSV)",          _wrap(ctx, effect_rainbow_wave)),
        ("Energy Comets",              _wrap(ctx, effect_energy_comets)),
        ("Peak Hold Columns (HSV)",     _wrap(ctx, effect_peak_hold_columns)),
        ("Full Strip Pulse (Palette)",  _wrap(ctx, effect_full_strip_pulse)),
        ("Waterfall (Palette)",         _wrap(ctx, effect_waterfall)),
        ("Bass Ripple Pulse v2",        _wrap(ctx, effect_bass_ripple_pulse_v2)),
        ("Spectral Blade (Clean)",      _wrap(ctx, effect_spectral_blade)),
        ("Bass Center Bloom (Clean)",   _wrap(ctx, effect_bass_center_bloom)),
        ("Peak Dots (Sparse)",          _wrap(ctx, effect_peak_dots)),
        ("Centroid Comet (Clean)",      _wrap(ctx, effect_centroid_comet)),
        ("Beat Outward Burst",          _wrap(ctx, effect_beat_outward_burst)),
        ("Quantized Sections",          _wrap(ctx, effect_quantized_sections)),
        ("Clean Fire – Edge Forge v3",  _wrap(ctx, effect_clean_fire_edge_v3)),
        ("Clean Fire – Center Vent v3", _wrap(ctx, effect_clean_fire_center_v3)),
    ]
