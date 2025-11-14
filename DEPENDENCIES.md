# Dependency Management for Reproducibility

## Python Version

**Python 3.11.3** (specified in `.python-version`)

We recommend using `pyenv` to manage Python versions:
```bash
pyenv install 3.11.3
pyenv local 3.11.3
```

## Dependency Files

### `requirements.txt` (Main Dependencies - Pinned)
Contains direct project dependencies with **exact versions** (`==`).

**Usage:**
```bash
pip install -r requirements.txt
```

This file includes:
- Core dependencies (pandas, numpy, scikit-learn, mlflow)
- Data versioning (DVC)
- API serving (FastAPI, uvicorn)
- Testing (pytest)

**Total: 16 main dependencies**

### `requirements-lock.txt` (Complete Freeze)
Contains **all** dependencies (direct + transitive) with exact versions.

Generated with: `pip freeze > requirements-lock.txt`

**Usage (maximum reproducibility):**
```bash
pip install -r requirements-lock.txt
```

This file includes all resolved subdependencies.

**Total: 178 packages**

## Which One to Use?

### For Normal Development
```bash
pip install -r requirements.txt
```
- Faster
- Easier to maintain
- Sufficient for most cases

### For Exact Reproducibility
```bash
pip install -r requirements-lock.txt
```
- Guarantees exact versions of EVERYTHING
- Recommended for:
  - Production deployments
  - Audits
  - Certifications
  - Debugging incompatibilities

## Updating Dependencies

### Update requirements.txt
1. Manually change version in `requirements.txt`
2. Install: `pip install -r requirements.txt`
3. Regenerate lock: `pip freeze > requirements-lock.txt`
4. Test everything works: `pytest tests/`
5. Commit both files

### Update complete lock
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Regenerate lock with current versions
pip freeze > requirements-lock.txt

# Verify changes
git diff requirements-lock.txt
```

## Installation Verification

To verify dependencies are correctly installed:

```bash
# Check critical versions
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import mlflow; print(f'mlflow: {mlflow.__version__}')"

# Run tests
pytest tests/ -v
```

## Critical Dependencies for Reproducibility

The following dependencies are critical for reproducible results:

| Package | Version | Reason |
|---------|---------|--------|
| numpy | 2.3.4 | Random operations, linear algebra |
| scikit-learn | 1.7.2 | ML algorithms, preprocessing |
| imodels | 2.0.3 | RuleFitRegressor implementation |
| pandas | 2.3.3 | Data manipulation |
| mlflow | 3.6.0 | Model tracking and versioning |

⚠️ **Changing these versions may result in different models.**

## Compatibility

### Tested Operating Systems
- ✅ macOS (Darwin 22.6.0)
- ⚠️ Linux (not tested yet)
- ⚠️ Windows (not tested yet)

### Architectures
- ✅ x86_64 / AMD64
- ⚠️ ARM64 / Apple Silicon (may require adjustments)

## Troubleshooting

### Error: "No matching distribution found"
```bash
# Update pip
pip install --upgrade pip

# Reinstall with clean cache
pip install -r requirements.txt --no-cache-dir
```

### Error: Version conflicts
```bash
# Create clean environment
python -m venv .venv-clean
source .venv-clean/bin/activate
pip install -r requirements-lock.txt
```

### Error: Package compilation
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install
```

## Security Notes

- 🔒 Recommended to check vulnerabilities periodically:
  ```bash
  pip install safety
  safety check -r requirements.txt
  ```

- 🔄 Update critical security dependencies without breaking reproducibility:
  - Only update patches (e.g., 2.3.3 → 2.3.4)
  - Avoid minor/major version changes
  - Always regenerate requirements-lock.txt
  - Run full test suite

## References

- [Python Dependency Management Best Practices](https://packaging.python.org/guides/tool-recommendations/)
- [Reproducible Data Science](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html)
