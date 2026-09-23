#!/bin/bash

set -e
echo "=== Running Adversarial Static Analysis Check ==="
echo "Running strict mypy over the current environment..."
make lint
echo "=== Adversarial Check Completed Successfully ==="
