import time
import multiprocessing

def test(n):
    for k in range(n):
        print(k)
        time.sleep(1)

def test2(n, queue):
    sharedVar = queue.get()
    sharedVar = n**2
    queue.put(sharedVar)

sharedVar = 0

t_queue = time.time()
queue = multiprocessing.Queue()
t_queue = time.time() - t_queue

queue.put(sharedVar)

t_process = time.time()
p = multiprocessing.Process(target = test2, name = "Test", args = (10, queue))
t_process = time.time() - t_process

t_process2 = time.time()
p.start()
t_process2 = time.time() - t_process2

t = 0
while True:
    if t >= 5:
        p.terminate()
        p.join(10)
    if p.is_alive():
        print("Ta vivo")
        time.sleep(0.2)
        t += 0.2
    else:
        break

print(queue.get())
print(t_queue)
print(t_process)
print(t_process2)
# p.terminate()
# p.join()