"""
    A simple process pool.
    It can sequentially complete a list of cmds.
    Besides, it can:

    1. Run multiple cmds in parallel, but not all cmds at once.
    2. Launching new process when an old one terminates to keep maximum parallelization.

    In short, it can always run the number of given processes in parallel.
"""

import subprocess
import time
from typing import List

process_pools: List[subprocess.Popen] = []

cmds: List[str] = [
    '/usr/bin/python3 /home/scripts/joern_runner_test.py -step 2 -id 0',
    '/usr/bin/python3 /home/scripts/joern_runner_test.py -step 4 -id 1',
    '/usr/bin/python3 /home/scripts/joern_runner_test.py -step 1 -id 2',
    '/usr/bin/python3 /home/scripts/joern_runner_test.py -step 3 -id 3',
    '/usr/bin/python3 /home/scripts/joern_runner_test.py -step 3 -id 4',
]

pool_max_size = 3

for cmd in cmds[:pool_max_size]:
    p = subprocess.Popen(cmd, shell=True)
    process_pools.append(p)

p_ptr = pool_max_size
while len(process_pools) > 0:
    for i in range(len(process_pools)):
        if process_pools[i].poll() is not None:
            # print(f'Output: \n{process_pools[i].communicate()[1]}\n')
            process_pools.pop(i)
            # Only start new process when a old process terminates
            if p_ptr < len(cmds):
                print(f'Launching process {p_ptr} at pos {i}')
                p = subprocess.Popen(cmds[p_ptr], shell=True)
                process_pools.append(p)
                p_ptr += 1
            print(f'Len: {len(process_pools)}')
            # Every time the length of the process pool changed, break and re-traverse
            break
    # Avoid too frequent polling
    time.sleep(2)
print('\n\n[Main] All done')