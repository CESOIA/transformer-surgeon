"""docs/gen_tables.py: the site's dictionary tables match the code."""
import importlib.util
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_generated_tables_are_up_to_date():
    spec = importlib.util.spec_from_file_location("gen_tables", DOCS / "gen_tables.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    for name, text in gen.render().items():
        assert (DOCS / "generated" / name).read_text() == text, f"run python docs/gen_tables.py ({name})"
