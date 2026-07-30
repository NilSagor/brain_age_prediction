# Requirements Management

This project uses **pip-tools** for deterministic dependency management.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pip install -r requirements.txt` | Reproduce paper results exactly |
| `pip install -r requirements-dev.txt` | Development with all tools |
| `pip-compile` | Regenerate locked requirements |

## Files Explained

### `requirements.in` **EDIT THIS**
High-level dependencies with semantic versioning bounds. This is what you modify when adding features.

Example:
```txt
torch>=2.1.0,<2.2.0    # Need 2.1+ for compile(), <2.2 for stability
