"""Native-library helpers: control the order of shared libraries in the process's global symbol scope."""

import ctypes


def load_into_global_scope(*names: str) -> list[str]:
    """Open each shared library ``name`` (e.g. ``"libstdc++.so.6"``) with ``RTLD_GLOBAL`` so that its symbols
    win over libraries opened into the global scope later. A library already loaded is only promoted; one that
    cannot be found is skipped. Returns the names that were loaded or promoted."""
    done = []
    for name in names:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            done.append(name)
        except OSError:
            pass
    return done


__all__ = ["load_into_global_scope"]
