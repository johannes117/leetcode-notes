# Heap/Priority Queue

## Kth Largest Element in a Stream
Design a class to find the kth largest integer in a stream of values, including duplicates. E.g. the 2nd largest from [1, 2, 3, 3] is 3. The stream is not necessarily sorted.

Implement the following methods:

constructor(int k, int[] nums) Initializes the object given an integer k and the stream of integers nums.
int add(int val) Adds the integer val to the stream and returns the kth largest integer in the stream.

```python
# maintain a min heap with exactly k elements, where the smallest element in the heap is the kth largest element in the stream
# init: initialise k, and min_heap as a list. loop through each number in nums array and run the self.add method
# add method
# Add the new value to the heap.
# if we have more than k elements, remove the smallest one
# the smallest element in our heap is the kth largest overall. return the smallest value in heap using [0]
import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.min_heap = []

        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)

        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)

        return self.min_heap[0]
```

### Key Concepts
- Maintain a min heap with exactly k elements, where the smallest element in the heap is the kth largest element in the stream
- The init function simply initialises k, min_heap as a list, and calls the add method for every number in the input array
- The add method, pushes the value on to the heap, pops if the heap length is greater than k, and returns the smallest element in the heap. 