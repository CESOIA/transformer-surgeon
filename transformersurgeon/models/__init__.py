"""Model family exports.

Import each family defensively so optional dependencies in one model do not
prevent importing unrelated model families.
"""

from importlib import import_module


def _safe_star_import(module_name):
	try:
		module = import_module(f"{__name__}.{module_name}")
	except Exception:
		return []

	exported = getattr(module, "__all__", None)
	if exported is None:
		exported = [name for name in dir(module) if not name.startswith("_")]

	for name in exported:
		if hasattr(module, name):
			globals()[name] = getattr(module, name)

	return exported


__all__ = []
for _module in [
	"qwen2_c",
	"qwen2_vl_c",
	"qwen2_5_vl_c",
	"vit_c",
	"bert_c",
	"distilbert_c",
	"llama_c",
]:
	__all__.extend(_safe_star_import(_module))

del _module
del _safe_star_import
del import_module
