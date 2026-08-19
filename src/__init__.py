"""
Repository-level initialization helpers.

We add a small compatibility shim for NumPy 2.x where some third-party
extensions expect the private symbol ``numpy.core.umath._center`` to exist.
This is a no-op shim that avoids attribute errors when running in Colab
with NumPy 2.x. It intentionally keeps behavior minimal.
"""
from __future__ import annotations

try:
	# Prefer private import for compatibility
	from numpy.core import umath as _umath
except Exception:
	_umath = None

if _umath is not None and not hasattr(_umath, "_center"):
	def _center(x):
		# best-effort identity placeholder to satisfy callers that only
		# test for the existence of this attribute
		return x
	setattr(_umath, "_center", _center)

# Expose a small top-level version helper for quick checks
def numpy_has_center_shim() -> bool:
	return _umath is not None and hasattr(_umath, "_center")
