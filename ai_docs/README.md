# AI Documentation Directory

## Purpose

This directory contains **AI-generated documentation** for reference purposes. It is separate from user-maintained ground-truth documentation (which will be in a `docs/` folder).

**Key Principle**: Documentation here contains technical references, implementation details, and open research questions, but **NO assumptions or conclusions about experimental results**.

## Directory Structure

```
ai_docs/
├── README.md                          # This file
├── IMPLEMENTATION_REFERENCE.md        # Technical implementation details
├── USAGE_EXAMPLES.md                  # Command-line usage examples
├── RESEARCH_QUESTIONS.md              # Potential research questions (hypotheses only)
└── YYYY-MM-DD/                        # Session-specific subdirectories
    ├── SESSION_SUMMARY.md             # What was accomplished in this session
    └── CODE_CHANGES.md                # Detailed log of code changes
```

## File Descriptions

### Root Level Files

**IMPLEMENTATION_REFERENCE.md**
- Technical details of the PCA analysis implementation
- API reference for classes and functions
- Data flow and processing pipeline
- Extension guidelines
- No experimental results or assumptions

**USAGE_EXAMPLES.md**
- Command-line usage examples
- Batch processing scripts
- Programmatic usage examples
- Troubleshooting commands
- No claims about what results "should" look like

**RESEARCH_QUESTIONS.md**
- Potential research questions to investigate
- Hypotheses to test (not conclusions)
- Open questions about dimensionality, trajectories, etc.
- Framework for hypothesis testing
- Explicitly labeled as speculation, not findings

### Session Subdirectories

Each session gets a dated subdirectory (YYYY-MM-DD format) containing:

**SESSION_SUMMARY.md**
- What was accomplished in that session
- Design decisions made
- Testing results
- File organization changes
- Next steps (not results)

**CODE_CHANGES.md**
- Detailed log of code modifications
- New files created
- Modified functions/classes
- Import changes
- Line counts and git status

## Organization Principles

### What Goes Here
✓ Implementation details and technical reference
✓ Usage instructions and examples
✓ Research questions and hypotheses
✓ Session logs and code change tracking
✓ API documentation

### What Does NOT Go Here
✗ Experimental results
✗ Conclusions about what patterns mean
✗ Assumptions about expected outcomes
✗ Claims about dimensionality or structure
✗ Ground-truth validated findings

## Relationship to User Documentation

- **ai_docs/** (this directory): AI-generated reference, technical details, open questions
- **docs/** (future, user-maintained): Ground-truth documentation, validated findings, experimental results

The user maintains `docs/` separately to ensure clear distinction between:
1. AI-generated technical reference (here)
2. Human-validated experimental findings (in `docs/`)

## Session History

### 2025-10-06
- Implemented PCA analysis module (`pca_analysis.py`)
- Added `analyze` command to main.py
- Created initial documentation structure
- No experimental results yet - implementation only

## Using This Documentation

### For Future Sessions
1. Read SESSION_SUMMARY.md from previous sessions to understand what was done
2. Check CODE_CHANGES.md to see exactly what code was modified
3. Refer to IMPLEMENTATION_REFERENCE.md for technical details
4. Use USAGE_EXAMPLES.md for command syntax

### For Development
1. IMPLEMENTATION_REFERENCE.md explains how the code works
2. CODE_CHANGES.md shows modification history
3. RESEARCH_QUESTIONS.md suggests directions for investigation

### For Research
1. RESEARCH_QUESTIONS.md outlines potential hypotheses
2. SESSION_SUMMARY.md documents progress
3. User creates separate `docs/` for validated findings

## Important Notes

1. **No assumptions**: This documentation does not make claims about what results will look like
2. **Technical focus**: Emphasis is on how the implementation works, not what it will discover
3. **Open questions**: Research questions are clearly labeled as hypotheses, not conclusions
4. **Session tracking**: Each session is logged separately for progress tracking
5. **User separation**: User maintains separate ground-truth documentation

## Future Sessions

For each new session:
1. Create new dated subdirectory (YYYY-MM-DD)
2. Log session summary and code changes
3. Update root-level docs if implementation changes
4. Keep assumptions out - focus on facts

## Meta

**Created**: 2025-10-06
**Purpose**: Organize AI-generated documentation separate from user documentation
**Maintenance**: AI updates this directory; user maintains `docs/`
