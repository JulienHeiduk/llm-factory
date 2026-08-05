# Data

Datasets used by the tutorials. Both subfolders are tracked but empty — the current tutorials all use inline example data, so nothing lives here yet.

| Folder | Contents |
|---|---|
| `raw/` | Datasets exactly as downloaded — never edited in place |
| `processed/` | Cleaned or prepared datasets derived from `raw/` |

The rule is that `raw/` is reproducible-by-download and `processed/` is reproducible-by-script, so neither needs to be precious.

**Do not commit large files.** Nothing here is gitignored today, so add a rule to [`.gitignore`](../.gitignore) before dropping in anything sizeable.
