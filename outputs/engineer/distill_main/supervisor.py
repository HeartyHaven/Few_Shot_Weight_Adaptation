"""Restart only this watchdog on unexpected exit; stop after completion or STOP."""
import fcntl
import os
import subprocess
import time
from common import HERE, PYTHON, ROOT, environment, atomic_json

def main():
    lock=(HERE/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(HERE/'supervisor_process.json',{'pid':os.getpid(),'started':time.time()})
    while not (HERE/'STOP').exists() and not (HERE/'COMPLETE.json').exists():
        p=subprocess.Popen([str(PYTHON),str(HERE/'watchdog.py')],cwd=ROOT,env=environment())
        code=p.wait()
        print('watchdog exited',code,time.strftime('%F %T'),flush=True)
        if (HERE/'COMPLETE.json').exists() or (HERE/'STOP').exists(): break
        time.sleep(15)

if __name__=='__main__': main()
