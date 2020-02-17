import os
import time

while True:
    error_code = os.system("python rollout.py auto")
    if error_code != 0:
        print("Error code", error_code)
        time.sleep(5*60)

    time.sleep(0.5*60)