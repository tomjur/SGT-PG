import Queue
import time
import numpy as np
import multiprocessing


class RandomSegmentCollector:
    def __init__(self, config, game_generation_function, collect_free):
        self.config = config
        self.results_queue = multiprocessing.Queue()

        self.collector_queues = [multiprocessing.JoinableQueue() for _ in range(self.config['general']['collectors'])]
        self.collectors = [
            CollectorProcess(config, self.results_queue, q, game_generation_function(), collect_free)
            for q in self.collector_queues
        ]

        for c in self.collectors:
            c.start()

    def generate(self, count):
        return [self.results_queue.get() for _ in range(count)]

    def close(self):
        for q in self.collector_queues:
            q.put(None)
        time.sleep(10.)
        for c in self.collectors:
            c.terminate()
        time.sleep(10.)


class CollectorProcess(multiprocessing.Process):
    def __init__(self, config, result_queue, collector_specific_queue, game, collect_free):
        multiprocessing.Process.__init__(self)
        self.result_queue = result_queue
        self.collector_specific_queue = collector_specific_queue
        self.config = config
        self.game = game
        self.collect_free = collect_free

    def run(self):
        required_size = self.config['model']['batch_size'] * self.config['general']['optimizations_per_cycle']
        while True:
            try:
                next_collector_specific_task = self.collector_specific_queue.get(block=True, timeout=0.1)
                task_type = next_collector_specific_task[0]
                # can only terminate
                self.collector_specific_queue.task_done()
                break
            except Queue.Empty:
                pass
            if self.result_queue.qsize() < required_size:
                for _ in range(100):
                    result = self._get_next()
                    self.result_queue.put(result)
            else:
                time.sleep(0.01)

    def _get_next(self):
        start = self._get_state()
        goal = self._get_state()
        is_collision = not self.game.check_terminal_segment((start, goal))
        cost = self._get_cost(start, goal, is_collision)
        return start, goal, cost

    def _get_state(self):
        if self.collect_free:
            return self.game.get_free_random_state()
        return self.game.get_random_state()

    # def _get_cost(self, start, goal, is_collision):
    #     distance = np.linalg.norm(start - goal)
    #     base_cost = self.config['cost']['base_cost']
    #     if is_collision:
    #         distance_coefficient = self.config['cost']['collision_cost']
    #         return distance_coefficient
    #     else:
    #         distance_coefficient = self.config['cost']['free_cost']
    #     return base_cost + distance_coefficient * distance

    def _get_cost(self, start, goal, is_collision):
        distance = np.linalg.norm(start - goal)
        base_cost = self.config['cost']['base_cost']
        if is_collision:
            distance_coefficient = self.config['cost']['collision_cost']
        else:
            distance_coefficient = self.config['cost']['free_cost']
        return base_cost + distance_coefficient * distance
