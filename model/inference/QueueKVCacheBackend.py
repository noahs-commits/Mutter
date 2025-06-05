import collections


class unordered_queue_backend:

    def __init__(self):
        self.min_id = 0
        self.max_id = 0

        self.arr=[]     
        self.arr_rev = {}
    #will return a index to put the next value in the queue
    def put(self):
        output=len(self.arr)

        self.arr_rev[self.max_id] = output
        self.arr.append(self.max_id)
        self.max_id += 1

        return output
    #return a eliment of the form (from_index,to_index) or remove_index 
    #if in the first form an eliment should be copied from the first index to the second index
    #if in the second form the eliment should be removed from the remove_index
    def pop(self):

        
        from_index=len(self.arr)-1
        to_index=self.arr_rev[self.min_id]

        if from_index==to_index:
            #if the last eliment is the one we want to remove
            #we can just remove it
            del self.arr_rev[self.arr.pop()]
            self.min_id += 1
            return from_index




        last_id=self.arr.pop()
        self.arr[to_index]=last_id


        self.arr_rev[last_id] = to_index
        del self.arr_rev[self.min_id]
        self.min_id += 1

        return from_index,to_index
    def check_invariants(self):
        assert len(self.arr)==len(self.arr_rev)
        assert len(self.arr)==self.max_id-self.min_id

        for index,value in enumerate(self.arr):
            assert value in self.arr_rev
            assert self.arr_rev[value]==index
    def __len__(self):
        return self.len

if __name__ == "__main__":
    from queue import Queue
    import random


    class unordered_queue:
        def __init__(self,max_size=1000):

            self.arr=[-1]*max_size
            self.back_end=unordered_queue_backend()
        def put(self,value):

            new_index=self.back_end.put()
            self.arr[new_index]=value
            return new_index
        def get(self):
            match self.back_end.pop():
                case (from_index,to_index):
                    output=self.arr[to_index]
                    self.arr[to_index]=self.arr[from_index]
                    self.arr[from_index]=-1
                case remove_index:
                    output=self.arr[remove_index]
                    self.arr[remove_index]=-1
            return output
        def __len__(self):
            return len(self.back_end.arr)
        

    #fuzz the unordered queue against a normal queue
    queue=unordered_queue(1000)
    truth_queue=Queue()
    for _ in range(1000):
        if bool(random.randint(0, 1)):
            value=random.randint(0, 1_000)
            queue.put(value)
            truth_queue.put(value)
        else:
            if len(queue)>0:
                assert queue.get()==truth_queue.get()
        queue.back_end.check_invariants()

        #check that the queue is the same except for the order
        assert collections.Counter(queue.arr[:len(queue)])==collections.Counter(truth_queue.queue), f"queue: {queue.arr[:len(queue)]} truth_queue: {list(truth_queue.queue)}"
