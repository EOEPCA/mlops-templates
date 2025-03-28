#!/usr/bin/env python3
import subprocess

subprocess.call(["git", "init", "."])
subprocess.call(["git", "add", "."])
subprocess.call(["git", "commit", "-m", "{{ cookiecutter.git_commit }}"])

print("Generated ML model '{{ cookiecutter.model_name }}' (TensorFlow)")
