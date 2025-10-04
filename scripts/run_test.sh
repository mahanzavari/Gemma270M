#!/bin/bash

# Script to run the test suite
echo "Running tests..."

# Set Python path to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run pytest on the tests directory
python -m pytest tests/ -v