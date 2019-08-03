import os
import time
import subprocess

#
# n_before = len(os.listdir('params/'))
# cmd = 'python3 localrunner.py -p1 "python3 strategy.py" -p2 "python3 ../examples/python_strategy.py"  -p3 "python3 ../examples/python_strategy.py" -p4 "python3 ../examples/python_strategy.py" -p5 "python3 ../examples/python_strategy.py" -p6 "python3 ../examples/python_strategy.py" '
# args = [f'-p{i} simple_bot' for i in range(2, 7)]
#
# while True:
#     prcss = subprocess.Popen(['python3', 'localrunner.py', '-p1 "python3 strategy.py"', args[0], args[1], args[2], args[3], args[4]], stdout=subprocess.PIPE)
#     while len(os.listdir('params/')) == n_before:
#         pass
#     n_before = len(os.listdir('params/'))
#     time.sleep(1)
#     prcss.kill()

i = 0
while True:
    i += 1
    os.system('python3 localrunner.py -p1 "python3 strategy.py" -p2 simple_bot -p3 simple_bot -p4 simple_bot -p5 simple_bot -p6 simple_bot --no-gui')
    time.sleep(0.2)
    os.system('find . -name "*.log.gz" -exec rm -f {} \;')
    if i % 100 == 0:
        print(i)