#!/bin/sh

# if [ -z "$1" ]
# then
#   echo "please enter screenName"
# else
#   # echo "it is running screen"
#   if [ -z "$STY" ]; then exec screen -dm -S screenName /bin/bash "$0"; fi
#   python SimRunner_0.py
# fi



if [ -z "$STY" ]; then exec screen -dm -S "$1" /bin/bash "$0"; fi
python SimRunner_0.py
