# Instructions

To reproduce the results in this study run

> python worker.py cuda:0

You can then monitor the progress in a separate shell using

> watch -n 5 python runner.py

Multiple instances of worker.py can be run with different devices. It is usually best to use two instances per device.

Specific experiments in dna_experiments.py can be commented out or reprioritised if needed. 
