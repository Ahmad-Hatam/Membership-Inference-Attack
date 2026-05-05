#!/bin/bash

# 1. Install missing dependencies inside the container
# We use -q to keep the logs clean and --no-cache-dir to save space
/opt/conda/bin/python -m pip install --no-cache-dir scikit-learn pandas requests scipy

# 2. Run your classifier
/opt/conda/bin/python -u meta_classifier.py