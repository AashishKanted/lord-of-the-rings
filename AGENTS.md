# Assignment repository

Treat `spec/customer_cell.json`, the customer packet, acceptance scenarios,
integration contract and `src/advaitic_systems/` as the common starter. Put your
robot solution in separate modules and configuration files. Report suspected
starter defects; do not silently change the customer specification or checker
to improve a scored result.

Robot `ASSEMBLY_DONE` and customer `ACCEPTED` are different events. Keep the
plant's measurements independent of robot decisions. A passing `systems-check`
trace shows log consistency, not a physical robot success. Do not describe test
fixtures or generated traces as completed robot runs.

Use the committed `uv.lock` for starter checks:

```sh
uv sync --locked --group dev --group physics
uv run pytest -q
uv run ruff check .
uv run systems-timing
uv run python -m advaitic_systems.parts
```
