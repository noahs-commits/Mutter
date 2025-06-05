from model.inference.QueueKVCacheBackend import unordered_queue_backend


class unordered_kvcache_queue:
    def __init__(self, max_size=1000):
        self.arr = [-1] * max_size
        self.back_end = unordered_queue_backend()
        self.max_size = max_size

    def put(self, value):
        new_index = self.back_end.put()
        self.arr[new_index] = value
        return new_index

    def get(self):
        match self.back_end.pop():
            case (from_index, to_index):
                output = self.arr[to_index]
                self.arr[to_index] = self.arr[from_index]
                self.arr[from_index] = -1
            case remove_index:
                output = self.arr[remove_index]
                self.arr[remove_index] = -1
        return output