"""
Top like utility for slurm
"""
import subprocess

cmd = "scontrol show nodes "
output = subprocess.check_output([x for x in cmd.split(" ")])

current_node = None
for line in output.split("\n"):
    if line.startswith("NodeName"):
        current_node = line.split(" ")[0].split("=")[1]
