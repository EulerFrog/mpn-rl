#!/bin/bash
# Generate architecture diagrams from Mermaid source files
#
# Prerequisites:
#   npm install -g @mermaid-js/mermaid-cli
#
# Usage:
#   ./generate_diagrams.sh

set -e  # Exit on error

echo "Generating architecture diagrams..."

# Diagram 1: Architecture Comparison (2400x1800)
echo "  [1/4] Generating architecture comparison..."
mmdc -i 1_architecture_comparison.mmd \
     -o 1_architecture_comparison.png \
     -b transparent \
     -w 2400 \
     -H 1800

# Diagram 2: Training Flow Comparison (2400x1800)
echo "  [2/4] Generating training flow comparison..."
mmdc -i 2_training_flow_comparison.mmd \
     -o 2_training_flow_comparison.png \
     -b transparent \
     -w 2400 \
     -H 1800

# Diagram 3: Problem and Solution (2800x1600)
echo "  [3/4] Generating problem and solution..."
mmdc -i 3_problem_and_solution.mmd \
     -o 3_problem_and_solution.png \
     -b transparent \
     -w 2800 \
     -H 1600

# Diagram 4: Implementation Plan (2000x1800)
echo "  [4/4] Generating implementation plan..."
mmdc -i 4_implementation_plan.mmd \
     -o 4_implementation_plan.png \
     -b transparent \
     -w 2000 \
     -H 1800

echo ""
echo "✓ All diagrams generated successfully!"
echo ""
ls -lh *.png
