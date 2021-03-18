import os
import time
import sys

while True:
    if len(sys.argv) == 2:
        error_code = os.system(f"python runner.py auto {sys.argv[1]}")
    else:
        error_code = os.system("python runner.py auto")
    if error_code != 0:
        print("Error code", error_code)
        time.sleep(5*60)

    time.sleep(0.5*60)