# DNA

Code for the paper **DNA: Proximal Policy Optimization with a Dual Network Architecture** published at NeurIPS 2022.

Please see the main branch for a more up-to-date version of the (ever-improving) algorithm.

# Instructions

To reproduce the results in this study run

> python worker.py cuda:0

You can then monitor the progress in a separate shell using

> watch -n 5 python runner.py

Multiple instances of worker.py can be run with different devices. It is usually best to use two instances per device.

Specific experiments in dna_experiments.py can be commented out or reprioritised if needed. 

## Citation
Please cite using the following bibtex entry:

```
@article{aitchison2022dna,
  title={DNA: Proximal Policy Optimization with a Dual Network Architecture},
  author={Aitchison, Mathew and Sweetser, Penny},
  journal={arXiv preprint arXiv:2206.10027},
  year={2022}
}
```
