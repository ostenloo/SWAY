#!/bin/bash
cd ~/SWAY/sway_harness
# wait for the current b2/b4/b6 build to finish.
# [m]ain regex avoids pgrep matching its own command line (the bug that hung the last waiter).
while pgrep -f "[m]ain.py build-all" >/dev/null; do sleep 60; done
echo "=== b2/b4/b6 done at $(date); starting b1 b3 b5 ==="
python main.py build-all --ids b1 b3 b5
echo "=== b1/b3/b5 ALL DONE at $(date) ==="
