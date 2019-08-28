import multiprocessing as mp
import sys
import os
from collections import defaultdict


class Worker(mp.Process):
    def __init__(self, task_queue, result_queue, worker_id, func, logger, passed_with_key=False, debug=False,
                 testing_mode=None, progress_freq=1000,batch_mode = False):
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.testing = testing_mode
        self._debug = debug
        self._id = worker_id
        self.func = func
        self.done = False
        self.logger = logger
        self.passed_with_key = passed_with_key
        self.progress_freq = progress_freq
        self.batch_mode = batch_mode

    def run(self):
        print("INFO: Worker {} is starting".format(self._id))
        self.done = False
        received = 0
        batch_count = 0
        while True:

            next_task = self.task_queue.get()

            if next_task is None:
                if self._debug:
                    print(self._id + " received poison pill")

                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            data = next_task
            if self.batch_mode:
                batch_count+=1
                for i in data:
                    received += 1
                    self._processData(i,received)
                print("INFO: Worker {} finished {} batches".format(self._id,batch_count))
            else:
                received+=1
                self._processData(data,received)
            self.task_queue.task_done()
        self.done = True
        print("INFO: Worker {} finished".format(self._id))

    def _processData(self, d,received):

        # Apply the function onto data
        try:
            result = self.func(d)
        except Exception as e:
            self.logger.error("Error raised on job {}".format(received))
            self.logger.error("data type: {}".format(type(d)))

            if isinstance(d, list):
                for i, v in enumerate(d):
                    self.logger.error("data[{}] = {}".format(i, d[i]))
            self.logger.exception(e)
            raise e

        # If the returned result has a key separate from the results, add both to the result queue
        if self.passed_with_key:
            key, res = result
            self.result_queue.put([key, res])
        else:
            self.result_queue.put(result)