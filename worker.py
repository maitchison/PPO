import os
import time

while True:
    error_code = os.system("python runner.py auto")
    if error_code != 0:
        print("Error code", error_code)

    time.sleep(120)