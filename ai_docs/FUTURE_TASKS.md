# Future Tasks

This document tracks potential future improvements and tasks for the mpn_rl project.

## Build System

### Add Hatchling Build Backend
**Status**: Not started
**Priority**: Low
**Date added**: 2025-10-06

**Description**:
Currently the project uses `pyproject.toml` for dependency management only. Consider adding a build backend (e.g., hatchling) to make the package installable as a proper Python package.

**Benefits**:
- Enables `pip install -e .` style installation
- Allows publishing to PyPI if desired
- Better integration with Python packaging ecosystem
- Supports proper versioning and metadata

**Changes needed**:
1. Add build backend to `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"
   ```

2. Consider project structure:
   - Move code into `src/mpn_rl/` directory (src layout)
   - Or keep flat layout (current)

3. Update README with installation instructions

**Considerations**:
- Current flat layout works fine for research/development
- Build backend mainly useful for distribution
- Can defer until/unless needed for publication or sharing

## Code Quality

### Type Hints
**Status**: Partial
**Priority**: Medium
**Date added**: 2025-10-06

**Description**:
Add comprehensive type hints to all functions and methods.

**Current status**:
- Some functions have type hints
- Many are missing return type annotations
- Consider using mypy for type checking

### Documentation

**Status**: Ongoing
**Priority**: Medium
**Date added**: 2025-10-06

**Description**:
- Add more detailed docstrings
- Create API reference documentation
- Add more usage examples

## Testing

### Unit Tests
**Status**: Not started
**Priority**: Medium
**Date added**: 2025-10-06

**Description**:
Add pytest-based unit tests for core functionality.

**Modules needing tests**:
- `mpn_module.py` - MPN layer behavior
- `mpn_dqn.py` - Q-value computation
- `pca_analysis.py` - PCA computations
- `model_utils.py` - Checkpoint save/load

### Integration Tests
**Status**: Not started
**Priority**: Low
**Date added**: 2025-10-06

**Description**:
Test full training pipelines end-to-end.

## Features

### Additional Analysis Tools
**Status**: Ideas stage
**Priority**: Low
**Date added**: 2025-10-06

**Potential additions**:
- Fixed point analysis for hidden states
- Temporal evolution of PCA structure during training
- Comparison plots across multiple agents
- State-space partitioning analysis
- M matrix structure analysis (sparsity, rank, etc.)

### Multi-Environment Support
**Status**: Partial (CartPole only for PCA)
**Priority**: Medium
**Date added**: 2025-10-06

**Description**:
Extend PCA analysis to support more environments:
- Add color extraction functions for other Gym environments
- Make state feature selection more flexible
- Support continuous action spaces

### Performance Optimizations
**Status**: Not started
**Priority**: Low
**Date added**: 2025-10-06

**Ideas**:
- Batch PCA computation for large datasets
- Parallel episode collection for analysis
- GPU-based PCA (cuML)
- Memory-efficient data collection

## Infrastructure

### CI/CD
**Status**: Not started
**Priority**: Low
**Date added**: 2025-10-06

**Potential setup**:
- GitHub Actions for testing
- Automatic linting (ruff, black)
- Test coverage reporting
- Documentation building

### Docker Support
**Status**: Not started
**Priority**: Low
**Date added**: 2025-10-06

**Description**:
Create Dockerfile for reproducible environment.

**Benefits**:
- Consistent environment across machines
- Easy GPU support
- Simplified deployment

## Research Directions

See `RESEARCH_QUESTIONS.md` for potential research investigations.

## Notes

- Tasks are prioritized as: High, Medium, Low
- Status options: Not started, In progress, Partial, Complete
- Add date when task is added
- Update status as work progresses
