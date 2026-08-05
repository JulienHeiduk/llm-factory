# Configs

Model and experiment configurations. **Empty so far** — every current tutorial hardcodes its settings (model name, temperature, thresholds) at the top of its script, which keeps each one readable standalone.

This folder is for when that stops scaling: shared YAML/TOML for sweeps, or one config per experiment run.

Python dependencies do **not** belong here — they live in [`pyproject.toml`](../pyproject.toml) as one dependency group per tutorial.
