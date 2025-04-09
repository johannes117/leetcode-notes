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


## Last Stone Weight
You are given an array of integers stones where stones[i] represents the weight of the ith stone.

We want to run a simulation on the stones as follows:

At each step we choose the two heaviest stones, with weight x and y and smash them togethers
If x == y, both stones are destroyed
If x < y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
Continue the simulation until there is no more than one stone remaining.

Return the weight of the last remaining stone or return 0 if none remain.

```python
# We need to use a max heap (a negated min heap)
# loop through each stone in the list, negate them (-)
# use heapify to turn it into a "max" heap
# while the length of the heap is above 1
# extract the two heaviest stones. (remember to negate to get the actual weights)
# If they are not equal, calculate the remaining weight and put it back (heappush). don't forget to negate
# return the wieght of the last stone, or 0 if none remain. 
import heapq

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        max_heap = [-stone for stone in stones]
        heapq.heapify(max_heap)

        while len(max_heap) > 1:
            first = -heapq.heappop(max_heap)
            second = -heapq.heappop(max_heap)

            if first != second:
                heapq.heappush(max_heap, -(first - second))
        
        if max_heap:
            return -max_heap[0]
        else:
            return 0
```

### Key Concepts:
- We need to use a max heap to solve this problem. A max heap is just a negated min heap. 
- we get the max_heap by negating each stone in the stones array and calling heapify. 
- While our max heap is greater than 1, we want to pop the two largest values
- When we pop values, we need to negate them to turn them into their original values again. 
- Now we can check if they are not equal, we want to put back the difference of the two stones on to the heap. We need to negate this value again to maintain the properties of the max heap. 
- If there is 1 stone left in the heap, we want to return it, remembering to negate it again. 
- Else we return 0. 


## K Closest Points to Origin
You are given an 2-D array points where points[i] = [xi, yi] represents the coordinates of a point on an X-Y axis plane. You are also given an integer k.

Return the k closest points to the origin (0, 0).

The distance between two points is defined as the Euclidean distance (sqrt((x1 - x2)^2 + (y1 - y2)^2)).

You may return the answer in any order.

```python
# Need to use a Max Heap so that we pushpop the kth largest tuple from the heap
# define a helper function to compute the distance from (0,0), simply return x**2 + y**2
# initialise a heap
# loop through x and y in points array
# calculate the distance using helper function
# if the heaplength is less than k, we want to pushpop, else just push. 
# dont forget to negate the distance when pushing
# at the end we need to return a list of x, y values that are left in the heap. 
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        def dist(x, y):
            return x**2 + y**2
        
        heap = []
        for x, y in points:
            d = dist(x, y)

            if len(heap) < k:
                heapq.heappush(heap, (-d, x, y))
            else: 
                heapq.heappushpop(heap, (-d, x, y))

        return [(x,y) for d, x, y in heap]
```

### Key Concepts
- We can use a Max Heap to maintain a list of the closest points to 0. 
- We use a max heap, because when we do a pushpop, its going to push the current point on to the heap, and pop the point thats furthest from 0. 
- We use a helper function to calculate the distance: d = x**2 + y**2. This is simplified for this usecase. 
- We need to make sure to negate the distance when pushing to the heap, this is to make the heap act like a max heap.

### Time and Space Complexity:
- Time: O(n log k), n because we loop through n points, log k because we push and pop from a heap of at most size k.
- Space: O(k), because we are storing at most k many things in the heap. 

## Kth Largest Element in an Array
Given an unsorted array of integers nums and an integer k, return the kth largest element in the array.

By kth largest element, we mean the kth largest element in the sorted order, not the kth distinct element.

Follow-up: Can you solve it without sorting?

```python
# Maintain a min heap of size k
# after processing all elements, the root of the heap will be the kth largest element
# initialise a heap
# loop through input array
# push the num onto the heap
# if the length of the heap exceeds k, heappop
# return the root of the heap [0]
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []

        for num in nums:
            heapq.heappush(heap, num)

            if len(heap) > k:
                heapq.heappop(heap)
            
        return heap[0]
```

### Key Concepts:
- We can use a min-heap to do this without sorting, in O(n log k) Time. 
- For each num in nums, push to heap, if heap larger than k, pop from heap. return the root of the heap.

### Time and Space:
- Time: O(n log k), loop through n values, pop and push is O(log k)
- since we're doing O(log k) work for each of the n elements, the total time complexity is O(n log k)
- Space: O(k): Heap never grows larger than k elements. 