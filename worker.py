"""
Jobs can be processed in parallel by running this script concurrently.
"""

import os
import time
import sys
import subprocess

while True:
    additional_args = sys.argv[1:]

    p = subprocess.Popen(
        ["python", "runner.py", "auto", *additional_args],
    )
    p.communicate()

    # # this hides errors...
    # p = subprocess.Popen(
    #     ["python", "runner.py", "auto", *additional_args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    # )
    # outs, errs = p.communicate()
    # outs = outs.decode("utf-8").split("\n")
    # errs = errs.decode("utf-8").split("\n")
    # if len(errs) > 0:
    #     print("Error:", errs)
    #     time.sleep(5*60)

    time.sleep(0.5*60)
