import numpy as np
import time

EXEC_TIME = 5
FREQ = 200
DT = 1/FREQ

if __name__ == "__main__":
    start = time.time()
    t = 0.0

    # Print "ROB311 @UM-ROBOTICS" for 5 seconds @200Hz

    p = 0
  
    while p < 5:
        print('ROB311 @UM-ROBOTICS')
        p = time.time() - start
        time.sleep(DT - ((p) %DT))

        