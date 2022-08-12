import threading
import time
import logging
import os
import subprocess


logging.basicConfig(filename="run_subprocess.log", filemode='w',
                    level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

gpu_list = [0, 1, 2, 3]
total_trials = 5

def wait_for_task(e, gpu, message):
    logging.debug('wait_for_task starting')
    time.sleep(5)

    while not e.isSet():
        if len(message) > 0:
            task, trial = message.pop(0)
            n_task = f'CUDA_VISIBLE_DEVICES={gpu} ' + task + f' --memo=T{trial}'
            logging.debug(f'({trial}/{(len(message)+1)}) ' + n_task)
            os.system(n_task)
        time.sleep(1)


if __name__ == '__main__':
    e = threading.Event()

    logging.debug('reading process list')
    with open('run_subprocess_list.txt') as fp:
        tasks = [_.strip() for _ in fp]
    
    thread_list = {}
    message = []
    for n in gpu_list:
        t = threading.Thread(name=f'blocking_{n}', 
                        target=wait_for_task,
                        args=(e, n, message,))
        thread_list[n] = t
        t.start()

    for trial in range(total_trials):
        for task in tasks:
            message.append((task, trial))

    for n in gpu_list:
        thread_list[n].join()

    e.set()
