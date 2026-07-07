# Deprecated tests

Everything under this directory predates the `test/unit` / `test/e2e` hardening
pass (see `FRAMEWORK_PROBLEMS.md`, section T*) and is kept for reference only —
it is **not** part of the default `pytest` run (`pytest.ini` sets
`testpaths = test/unit test/e2e`).

Most files here are not actually pytest tests: they are `argparse` + `main()` CLI
scripts (`test_*` functions that take a positional `args`), several share a
basename with a file in another legacy directory (`inference_test.py`,
`test_messages.py`) which breaks collection when run together, and some call
stale APIs (e.g. `manager.set("pruning", ...)` instead of
`"structured_pruning"`/`"unstructured_pruning"`).

Do not add new tests here. New coverage belongs in `test/unit` (fast, no
downloads) or `test/e2e` (real models / real export backends, capability-gated
via `test/_helpers/capabilities.py`). If a script here still has unique coverage
worth keeping, port it into `test/e2e` as a proper parametrized/marked pytest
test and delete the original rather than maintaining both.
