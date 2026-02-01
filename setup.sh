#!/bin/bash

echo " Initializing ATLAS-X Environment..."

# 1. Create missing __init__.py files for package discovery
touch src/__init__.py
touch src/utils/__init__.py
touch src/app/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py

# 2. Install/Update dependencies
pip install -r requirements.txt

# 3. Set Python Path for the current session
export PYTHONPATH=$PYTHONPATH:.

echo " Environment Ready. To start, run: streamlit run src/app/dashboard.py"