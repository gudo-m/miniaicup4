import os
import time
import subprocess


cmd = 'python3 localrunner.py -p3 "python3 strategy.py"  -p1 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot -p2 simple_bot --no-gui'
cmd2 = 'python3 localrunner.py -p1 "python3 strategy.py" -p2 simple_bot -p3 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot --no-gui'

i = 0
while True:
    i += 1
    os.system(cmd)
    time.sleep(0.2)
    os.system('find . -name "*.log.gz" -exec rm -f {} \;')
    if i % 25 == 0:
        print(i)