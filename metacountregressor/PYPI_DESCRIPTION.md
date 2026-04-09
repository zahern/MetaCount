# metacountregressor

JAX-first hierarchical search and fitting for count, CMF, duration, and linear models.

The package now exposes:

- count-model search and manual fitting
- CMF search and manual fitting
- duration experiments on a JAX hierarchical lognormal path
- linear experiments on a JAX hierarchical Gaussian path
- latent-class probability extraction
- packaged example datasets

Bundled example data includes the Example 16-3 CSV and synthetic platform-speed examples.

Important packaging note:

- the long description shown on PyPI comes from the uploaded distribution metadata
- if PyPI still shows old text, you are almost certainly uploading stale files from `dist/` or building from an older checkout

Before upload:

```bash
Remove-Item -Recurse -Force dist, build, *.egg-info
python -m build
```

Then inspect the wheel metadata and confirm:

- version is the new version you intend to upload
- summary matches the current `pyproject.toml`
- the long description body matches the current README/cookbook
