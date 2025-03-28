#!/usr/bin/env python3
import sys

MLFLOW_TRACKING_URI = "{{ cookiecutter.mlflow_tracking_uri }}"

if not MLFLOW_TRACKING_URI:
    print("ERROR: mlflow_tracking_uri is empty.")
    sys.exit(-1)
