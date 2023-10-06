import os
import sys

print(sys.argv)
if len(sys.argv) == 1:
    print("please enter the screen name")
else:
    print("screenName is: ",  sys.argv[1])
    screenCommand = f"if [ -z \"$STY\" ]; then exec screen -dm -S { sys.argv[1]} /bin/bash \"$0\"; fi"

    os.system(screenCommand)
    os.system("python cfqr_scaled_x_isolated_trial.py")
    os.system("python zip_data.py")
