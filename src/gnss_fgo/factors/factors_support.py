"""Cross-cutting helpers shared between the DD factor builder and the
DDCP build policy. The substantial factor builders live in their own
modules (``imu_preintegration.py``, ``nhc.py``, ``doppler_sd.py``,
``zupt.py``) and are imported directly by their consumers.
"""

from ..utils import get_wavelengths as _utils_get_wavelengths


def get_wavelengths(tc, obs, sat):
    """Thin adapter — see utils.ambiguity.get_wavelengths."""
    return _utils_get_wavelengths(obs, sat, glo_ch=tc.nav.glo_ch)
