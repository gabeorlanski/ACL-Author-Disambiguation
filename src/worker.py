import multiprocessing as mp
import sys
import os
from collections import defaultdict


class Worker(mp.Process):
    def __init__(self, task_queue, result_queue, worker_id, func, debug=False, testing_mode=None):
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.testing = testing_mode
        self._debug = debug
        self._id = worker_id
        self.func = func
        self.done = False

    def run(self):
        self.done=False
        while True:

            next_task = self.task_queue.get()

            if next_task is None:
                if self._debug:
                    print(self._id + " received poison pill")

                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            data = next_task

            # Apply the getListings onto the
            key, result = self.func(data)

            self.result_queue.put([key, result])
            self.task_queue.task_done()
        self.done = True
        print("INFO: Worker {} finished".format(self._id))