#!/bin/bash

# Install only what is needed
/opt/conda/bin/python -m pip install --no-cache-dir pandas

# Run attack
/opt/conda/bin/python -u rmia.py