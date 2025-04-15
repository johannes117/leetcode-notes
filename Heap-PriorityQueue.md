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

## Task Scheduler
You are given an array of CPU tasks tasks, where tasks[i] is an uppercase english character from A to Z. You are also given an integer n.

Each CPU cycle allows the completion of a single task, and tasks may be completed in any order.

The only constraint is that identical tasks must be separated by at least n CPU cycles, to cooldown the CPU.

Return the minimum number of CPU cycles required to complete all tasks.

```python
# Each Task, 1 Unit of time
# Minimise Idle Time
# count the frequency of each task, store in hashmap (Counter)
# create a maxHeap using the negated counts. (you can use heapify)
# initialise a time counter, and a double ended queue (deque)
# while heap or queue are non-empty
# Increment the time
# if heap, pop from heap, decrement the value (we can actually add 1 since the values are negated)
# append the popped value to the queue, [val, time+n]
# if queue, and the top value in the queue's timestamp is equal to time:
# pop from queue and push to heap. 
# return the final time. 
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        
        taskCount = Counter(tasks)
        maxHeap = [-task for task in taskCount.values()]
        heapq.heapify(maxHeap) 

        time = 0
        queue = deque()

        while maxHeap or queue:
            time += 1
            if maxHeap:
                task = 1 + heapq.heappop(maxHeap) # Add one here to actually decrement the value
                if task:
                    # add that value to the queue
                    queue.append([task, time+n])
            
            if queue and queue[0][1] == time:
                heapq.heappush(maxHeap, queue.popleft()[0])
        return time
```

### Key Concepts
- We can use a max Heap and a Queue to track which task should be processed in what order. 
- The queue tracks the "Idle" time, by storing the task along with a timestamp
- When we reach that timestamp in the loop, we popleft that task from the queue and add it back to the heap. 
- The heap allows us to process the most frequent tasks first, this will allow us to find the least Interval time value. 

### Time and Space Complexity
- Time: O(n * m), where n is the number of tasks and m is the number of wait time "n".
Note: The queue will have at most 26 values, which would make accessing it O(log26), which in this case is essentially constant time. 
- Space: O(n), we are storing atleast n values in a heap.


## Design Twitter
Implement a simplified version of Twitter which allows users to post tweets, follow/unfollow each other, and view the 10 most recent tweets within their own news feed.

Users and tweets are uniquely identified by their IDs (integers).

Implement the following methods:

Twitter() Initializes the twitter object.
void postTweet(int userId, int tweetId) Publish a new tweet with ID tweetId by the user userId. You may assume that each tweetId is unique.
List<Integer> getNewsFeed(int userId) Fetches at most the 10 most recent tweet IDs in the user's news feed. Each item must be posted by users who the user is following or by the user themself. Tweets IDs should be ordered from most recent to least recent.
void follow(int followerId, int followeeId) The user with ID followerId follows the user with ID followeeId.
void unfollow(int followerId, int followeeId) The user with ID followerId unfollows the user with ID followeeId.

```python
class Twitter:

    def __init__(self):
        # initialise time (timestamp), tweets as a dictionary, and following as a dictionary
        self.time = 0
        self.tweets = {}
        self.following = {}
        

    def postTweet(self, userId: int, tweetId: int) -> None:
        # initialise users data structures if they don't exist
        if userId not in self.tweets:
            self.tweets[userId] = []
        if userId not in self.following:
            self.following[userId] = set([userId])
        # add tweet with current timestamp and increment time. 
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1
        

    def getNewsFeed(self, userId: int) -> List[int]:
        # initialise user if they dont exist
        if userId not in self.tweets:
            self.tweets[userId] = []
        if userId not in self.following:
            self.following[userId] = set([userId])
        # collect all tweets from users that this user follows
        all_tweets = []
        for followee in self.following[userId]:
            if followee in self.tweets:
                all_tweets.extend(self.tweets[followee])
        # sort by time (most recent first) and take top 10
        all_tweets.sort(reverse=True)
        return [tweetId for time, tweetId in all_tweets[:10]]
        

    def follow(self, followerId: int, followeeId: int) -> None:
        # initialise user if they dont exist
        if followerId not in self.following:
            self.following[followerId] = set([followerId])
        if followerId not in self.tweets:
            self.tweets[followerId] = []
        # add followee to followers following set
        self.following[followerId].add(followeeId)        

    def unfollow(self, followerId: int, followeeId: int) -> None:
        # can't unfollow yourself and can only unfollow if following exists
        if followerId == followeeId or followerId not in self.following:
            return
        # remove followee from follower's following set if they exist. 
        self.following[followerId].discard(followeeId)
```

### Key Concepts:
- Can use dictionaries to mimic a document database
- Can use sorting and array slicing to get the top 10 tweets. Probably should use a heap

### Time and Space Complexity
Time Complexity:
- postTweet: O(1)
- follow/unfollow: O(1)
- getNewsFeed: O(NlogN) where N is the total number of tweets from all followed users (due to sorting)

## Find Median From Data Stream
The median is the middle value in a sorted list of integers. For lists of even length, there is no middle value, so the median is the mean of the two middle values.

For example:

For arr = [1,2,3], the median is 2.
For arr = [1,2], the median is (1 + 2) / 2 = 1.5
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far.

```python
# SmallHeap (Max Heap)
# Large Heap (Min Heap)
# Median is either the top element of one heap (odd number of elements) or the average of the tops of both heaps
# addNumm
# check if heaps are equal
            # add to small, but go through large first (pushpop). negate the value that comes out of the large heap when inserting into the small heap
        # else
            # add to large, but go through small first (pushpop). negate the value that comes out of the small heap. pushpop using the negated num value
# findMedian:
 # if heaps are equal - even number of elements, average of the two
            # return the average of the two middle values
            # (get negated top from small plus top of large) divided by 2
        # else
            #  return the negated top of small. odd number of elements, top of small has the median. 

class MedianFinder:

    def __init__(self):
        # init the small and large heaps as lists
        self.small = []
        self.large = []

    def addNum(self, num: int) -> None:
        if len(self.small) == len(self.large):
            # Add to small, go through large
            heapq.heappush(self.small, -heapq.heappushpop(self.large, num))
        else:
            # add to large if uneven, go through small
            heapq.heappush(self.large, -heapq.heappushpop(self.small, -num))   

    def findMedian(self) -> float:
        if len(self.small) == len(self.large):
            # take the average of the middle values
            return (-self.small[0] + self.large[0]) / 2
        else:
            # take the top value of small
            return -self.small[0]       
```

### Key Concepts:
- If we maintain a small (max_heap) and a large (min_heap), 
- where the large heap contains values that are all greater than or equal to the values in the small heap
- Then we can take the smallest value in the large heap, and the largest value in the small heap and find the median value.
- If the heaps are unequal, then we simply take the largest value in the small heap. 
- This works because we always add to the small heap if the heap lengths are equal. 
- We use pushpop through the other heap first because this ensures we always maintain the small and large heap properties where values in small <= large

### Time and Space Complexity
Time: 
- addNum: O(logn) - Both heappush and heappushpop operations take O(logN) time where N is the number of elements in the heap. 
- findMedian: O(1)
Space:
- O(n): We store all N elements across the two heaps. 