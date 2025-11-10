# effects/__init__.py
from .basics import (
    effect_line_spectrum,
    effect_mirror_spectrum,
    effect_rainbow_wave,
)
from .dynamics import (
    effect_peak_hold_columns,
    effect_full_strip_pulse,
    effect_waterfall,
    effect_bass_ripple_pulse_v2,
)
from .clean import (
    effect_spectral_blade,
    effect_bass_pulse_core,
    effect_peak_dots_expanded,
    effect_bass_impact_wave,
)
from .fire import (
    effect_clean_fire_edge_v4,
    effect_clean_fire_center_v4,
)

def _wrap(ctx, fn):
    return lambda bands, beat, active, _f=fn: _f(ctx, bands, beat, active)

def build_effects(ctx):
    return [
        ("Line Spectrum (HSV)",                 _wrap(ctx, effect_line_spectrum)),
        ("Mirror Spectrum (HSV)",               _wrap(ctx, effect_mirror_spectrum)),
        ("Rainbow Wave (HSV)",                  _wrap(ctx, effect_rainbow_wave)),

        ("Peak Hold Columns (HSV)",             _wrap(ctx, effect_peak_hold_columns)),
        ("Full Strip Pulse (Palette)",          _wrap(ctx, effect_full_strip_pulse)),
        ("Waterfall (Palette)",                 _wrap(ctx, effect_waterfall)),
        ("Bass Ripple Pulse v2",                _wrap(ctx, effect_bass_ripple_pulse_v2)),

        ("Spectral Blade (Clean)",              _wrap(ctx, effect_spectral_blade)),
        ("Bass Pulse Core (NEW)",               _wrap(ctx, effect_bass_pulse_core)),
        ("Peak Dots Expanded",                  _wrap(ctx, effect_peak_dots_expanded)),
        ("Beat Impact Wave",                    _wrap(ctx, effect_bass_impact_wave)),

        ("Fire Edge Forge v4 (Palette + Sparks)", _wrap(ctx, effect_clean_fire_edge_v4)),
        ("Fire Center Vent v4 (Palette + Sparks)", _wrap(ctx, effect_clean_fire_center_v4)),
    ]