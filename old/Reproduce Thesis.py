"""
Running this scritp will reproduce all results for my (Matthew Aitchison's) thesis.
Be prepaired, this will take a long time.

Note: maybe this should be a container?
"""

import os

def install_prerequisites():
    pass

def run_notebook(notebook_path):
    os.system(f"jupyter nbconvert - -execute '{notebook_path}'"

def run_experiments():
    pass

def generate_plots():

    # Chapter 1

    # Chapter 2
    run_notebook("eRP - Replay Buffer.ipynb")