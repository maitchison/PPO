import os
import time

while True:

    # this just shows what jobs are pending, and what order they will be done in.
    os.system("python runner.py show")
    print()
    error_code = os.system("python runner.py auto")
    if error_code != 0:
        print("Error code", error_code)
        break

    time.sleep(120)