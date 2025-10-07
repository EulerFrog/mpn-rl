# Code Cleanup and Build Setup - October 6, 2025

## Objective
Clean up code for reproducible builds from clean git clone, remove conditional imports, and set up proper dependency management with uv.

## Changes Made

### 1. Created `pyproject.toml`
**File**: `/home/eulerfrog/KAM/mpn_rl/pyproject.toml`
**Purpose**: Dependency management with uv/pip

**Contents**:
```toml
[project]
name = "mpn-rl"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "gymnasium>=0.29.0",
    "scikit-learn>=1.3.0",
    "imageio>=2.31.0",
    "pillow>=10.0.0",
    "tensordict>=0.2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4.0", "black>=23.7.0", "ruff>=0.0.280"]
```

**Note**: No build backend (hatchling) included yet - deferred to future (see FUTURE_TASKS.md)

### 2. Removed Conditional Imports

#### File: `render_utils.py`
**Before**: Try/except blocks for matplotlib, PIL, imageio with availability flags
**After**: Direct imports at top

**Changes**:
- Lines 16-39: Removed try/except blocks, added direct imports
- Line 49: Removed `if not MATPLOTLIB_AVAILABLE` check
- Line 88: Removed `if not PIL_AVAILABLE` check
- Lines 137-142: Removed three availability checks
- Line 163: Removed `breakpoint()` call
- Line 350: Removed availability check in __main__
- Lines 364-382: Removed try/except around CartPole test

**Imports now at top**:
```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import imageio
```

#### File: `visualize.py`
**Before**: Try/except block for imageio in `visualize_episode()` function
**After**: Direct import at top

**Changes**:
- Line 17: Added `import imageio` to top imports
- Lines 324-329: Removed try/except block around imageio usage

### 3. Created `.gitignore`
**File**: `/home/eulerfrog/KAM/mpn_rl/.gitignore`
**Purpose**: Proper git exclusions

**Excludes**:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments
- IDE files
- Experiment outputs (except structure)
- Generated files (`.pt`, `.gif`, `.png`)
- Logs and temporary files

**Preserves**:
- Project structure
- Documentation images in ai_docs
- `.gitkeep` files for directory structure

### 4. Updated `README.md`

**Installation Section**:
- Added uv installation instructions (recommended)
- Added pip installation as alternative
- Listed all dependencies explicitly
- Removed fragmented installation options

**Usage Section**:
- Added `analyze` command documentation
- Added PCA analysis examples
- Updated experiment directory structure to include `analysis/`

**Files Section**:
- Added `pca_analysis.py` documentation

**Directory Structure**:
- Added `pyproject.toml`
- Added `.gitignore`
- Added `ai_docs/` directory
- Updated `main.py` description to include `analyze`

### 5. Created Future Tasks Documentation
**File**: `ai_docs/FUTURE_TASKS.md`
**Purpose**: Track potential future improvements

**Sections**:
- Build System (hatchling setup - deferred)
- Code Quality (type hints, documentation)
- Testing (unit tests, integration tests)
- Features (additional analysis tools, multi-environment)
- Infrastructure (CI/CD, Docker)

## Verification

### No Conditional Imports Remaining
```bash
$ grep -n "try:\s*import\|except ImportError" *.py
# No results - all conditional imports removed
```

### All Imports at Top
All external package imports now at module top, following Python best practices.

## Installation Workflow (New)

From clean git clone:
```bash
git clone <repo-url>
cd mpn_rl

# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .

# Ready to run
python3 main.py train
python3 main.py analyze --experiment-name <name>
```

## Benefits

**Before**:
- Fragmented installation instructions
- Conditional imports made dependencies unclear
- No single source of truth for requirements
- Manual dependency management

**After**:
- Single `pyproject.toml` defines all dependencies
- All imports required and explicit
- Clean installation from git clone
- Works with modern Python tools (uv, pip)
- No silent failures from missing packages

## Philosophy

**Removed**: "Optional" dependency approach with try/except
**Rationale**:
- Research code benefits from explicit requirements
- Silent fallbacks hide problems
- Installation from pyproject.toml ensures all deps available
- Clearer code without conditional logic

**Trade-off**: Larger dependency footprint, but appropriate for research codebase.

## Not Included (Deferred)

### Build Backend (Hatchling)
**Why deferred**:
- Current flat layout works fine for development
- Build backend mainly useful for PyPI publication
- Can add later if needed
- Documented in FUTURE_TASKS.md for future reference

**When to add**:
- If publishing to PyPI
- If distributing as installable package
- If want `pip install mpn-rl` from git

## Testing Checklist

- [ ] Fresh install with uv works
- [ ] Fresh install with pip works
- [ ] All modules importable
- [ ] No import errors from removed conditional imports
- [ ] Main CLI commands work
- [ ] PCA analysis works

## Files Modified

1. **Created**:
   - `pyproject.toml`
   - `.gitignore`
   - `ai_docs/FUTURE_TASKS.md`
   - `ai_docs/2025-10-06/CLEANUP_AND_BUILD.md` (this file)

2. **Modified**:
   - `render_utils.py` - Removed conditional imports (8 changes)
   - `visualize.py` - Removed conditional import (1 change)
   - `README.md` - Updated installation and usage sections

## Next Steps

1. Initialize git repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: MPN-RL with PCA analysis"
   ```

2. Test installation workflow on another machine

3. Document experimental findings separately (user to maintain `docs/`)

## Notes for Future Sessions

- All dependencies are now required (no optional packages)
- pyproject.toml is the single source of truth for dependencies
- Conditional imports removed - fail fast if package missing
- Build backend (hatchling) deferred - see FUTURE_TASKS.md
