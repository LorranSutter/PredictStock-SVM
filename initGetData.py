import os
import sys
import time
import signal
import subprocess
import datetime as dt

'''
Manage data acquiring by getData.py script
'''

db_dir = 'db'

def readPid():
    if os.path.exists(db_dir + '/pid.txt'):
        with open(db_dir + '/pid.txt', 'r') as f:
            pid = f.readline()
        return pid

def writePid(pid):
    with open(db_dir + '/pid.txt', 'w') as f:
        f.write(str(pid))

# Start processes
def startProcess(processName):
    command = "gnome-terminal -- python3 {0}".format(processName)
    grepCommand = "pgrep -lf 'python3 {0}'".format(processName)

    subprocess.Popen([command, "--disable-factory"], shell = True, preexec_fn = os.setpgrp)

    time.sleep(1)
    print("\n{0} process initilized!\n{1}".format(processName, dt.datetime.today()))

    # Exec subprocess to get pids
    pGrep = subprocess.Popen(grepCommand,stdout = subprocess.PIPE, shell = True)

    # Get processes names and its pids
    pidsProcess = pGrep.communicate()
    pidsProcess = pidsProcess[0].decode().split('\n')

    for element in pidsProcess:
        if 'python3' in element: # Get python3 pid
            return int(element.split()[0])


## * ---------- MAIN ---------- ##
if __name__ == "__main__":

    with open(db_dir + '/lastId.txt', 'r') as f:
        lastId = f.readline()
    if lastId == '-1':
        print("\nAlready finished!\n")

    # Processes names
    pid = readPid()

    try:
        os.kill(int(pid), signal.SIGSTOP) # Kill idle process
    except:
        pass
    
    pid = startProcess("getData.py")
    writePid(pid)
    
    print(pid)
    while True:
        if time.time() - os.stat(db_dir + '/pid.txt').st_mtime > 30:
            if all([time.time() - os.stat(db_dir + '/stocks' + '/' + d).st_mtime > 30 for d in os.listdir(db_dir + '/stocks')]):
                try:
                    os.kill(pid, signal.SIGSTOP) # Kill idle process
                except:
                    pass

                with open('lastId.txt', 'r') as f:
                    lastId = f.readline()
                if lastId == '-1':
                    print("\nFinished!\n")
                    break
                with open('lastId.txt', 'w') as f:
                    f.write(str(int(lastId) + 1))

                pid = startProcess("getData.py")
                print(pid)
                writePid(pid)