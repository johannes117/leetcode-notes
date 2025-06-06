## Maximum Subarray
Given an array of integers nums, find the subarray with the largest sum and return the sum.

A subarray is a contiguous non-empty sequence of elements within an array.

```python
# Can be solved using "Kadane's Algorithm"
# maintain a max_sum and curr sum
# loop through nums
# increment curr_sum with current number
# set max sum
# if curr_sum less than 0, reset it to 0
# return max_sum
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = float("-inf")
        curr_sum = 0

        for i in range(len(nums)):
            curr_sum += nums[i]
            max_sum = max(max_sum, curr_sum)

            if curr_sum < 0:
                curr_sum = 0
        
        return max_sum
```

### Key Concepts:
- Kadane's Algorithm: 
- Maintain 2 variables: 
max_so_far: The max bound so far (max_sum)
max_ending_here: the max sum of subarray ending at the current position (curr_sum)
- Key Insight: at each position, we decide whether to:
    - Start a new subarray from current element, or (when we reset curr_sum we are starting a new subarray)
    - Extend the existing subarray by including the current element. (when we don't reset curr_sum, we extend it)

### Time and Space:
- Time: O(n), single pass
- Space: O(1)

## Jump Game
You are given an integer array nums where each element nums[i] indicates your maximum jump length at that position.

Return true if you can reach the last index starting from index 0, or false otherwise.

```python
# return true if you can reach the last index starting from 0
# Greedy - Start at end
# initialise n and target variables (target is the last value in the list)
# loop backwards (start, stop, step)
# set max_jump to the value at current index
# if index + max jump atleast target, then set target to index. (move target pointer to current index)
# return if target pointer reached 0
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        target = n-1

        for i in range(n-1, -1, -1):
            max_jump = nums[i]
            if i + max_jump >= target: 
                target = i
        
        return target == 0
```

### Key Concepts:
- This can be solved using a Greedy approach
- Start at the end, and iterate backwards. 
- We use a target variable to check if the current index can reach the target. 
- If it does, we move the target backwards, resetting the "goal". 
- If the target pointer reaches 0, that means we were able to reach the end, and we should return true. 

### Time and Space:
- Time: O(n)
- Space: O(1)


## Jump Game II
You are given an array of integers nums, where nums[i] represents the maximum length of a jump towards the right from index i. For example, if you are at nums[i], you can jump to any index i + j where:

j <= nums[i]
i + j < nums.length
You are initially positioned at nums[0].

Return the minimum number of jumps to reach the last position in the array (index nums.length - 1). You may assume there is always a valid answer.

Example 1:

Input: nums = [2,4,1,1,1,1]

Output: 2
Explanation: Jump from index 0 to index 1, then jump from index 1 to the last index.

```python
# Return the minimum number of jumps to reach the last position
# Greedy Approach
# initialise smallest, n, end and far variables
# loop to n - 1 (prevents off by 1 error)
# set far pointer to the max of itself and i = nums[i]
# if i is at end pointer: increment smallest, and set end to far
# return smallest
class Solution:
    def jump(self, nums: List[int]) -> int:
        smallest = 0
        n = len(nums)
        end, far = 0, 0

        for i in range(n-1):
            far = max(far, i + nums[i])
            
            if i == end:
                smallest += 1
                end = far
        
        return smallest
```

### Key Concepts:
- Divide the array into "regions", each region has a smallest number of jumps that can reach that region. 
- When the index reaches the end pointer, we have reached the end of that region. 
- We need to set the end pointer to the far pointer, and increment the smallest count. 
- The far pointer will always be set to the furthest index we can get to from the current index. 

### Time and Space:
- Time: O(n)
- Space: O(1)

## Gas Station
There are n gas stations along a circular route. You are given two integer arrays gas and cost where:

gas[i] is the amount of gas at the ith station.
cost[i] is the amount of gas needed to travel from the ith station to the (i + 1)th station. (The last station is connected to the first station)
You have a car that can store an unlimited amount of gas, but you begin the journey with an empty tank at one of the gas stations.

Return the starting gas station's index such that you can travel around the circuit once in the clockwise direction. If it's impossible, then return -1.

It's guaranteed that at most one solution exists.


```python
# If valid, return the index of the starting gas station where we can complete an entire loop.
# Intuition: If we can make it to the end of the array, then we have found the valid starting point
# Edgecase: if sum of gas less than sum of cost, return 0
# init total and start variables
# loop through gas list
# increment the total by the difference between gas and cost at index i
# if total dips below 0, reset the state, and increment the start pointer to i + 1
# return start pointer
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        
        total = 0
        start = 0
        for i in range(len(gas)):
            total += (gas[i] - cost[i])

            if total < 0:
                total = 0
                start = i + 1
        
        return start
```

### Key Concepts
- Greedy Approach: Try every starting position, and move on the moment we detect it won't work
- We have a total counter that if it dips below 0, we disregard that cycle by resetting the total and moving the starting position. 
- If a starting position can make it to the end of the array, we know that that position is the one we are looking for
- We know this because we are guaranteed to have 1 and only 1 solution, and no matter what starting point, the loop must wrap around the end of the array for it to be valid
- If we can reach the end of the array from any starting point, we know that that must be the point we are looking for. 


### Time and Space
- Time: O(n)
- Space: O(1)

## Hand of Straights
You are given an integer array hand where hand[i] is the value written on the ith card and an integer groupSize.

You want to rearrange the cards into groups so that each group is of size groupSize, and card values are consecutively increasing by 1.

Return true if it's possible to rearrange the cards in this way, otherwise, return false.

```python
# return true if we can form groups of size groupSize
# if hand is not divisible by groupsize return false
# initialise count dictionary using Counter()
# initialise a min_heal variable using List(count.keys())
# heapify
# while heap
# get the mallest card from the heap
# loop through groupsize
# grab card by offsetting first by i
# if count is 0 return false
# decrement count, if card count is 0, pop, and check if its the smallest if not return false
# return true if the min_heap reaches 0
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False
        
        count = Counter(hand)
        min_heap = list(count.keys())
        heapq.heapify(min_heap)

        while min_heap:
            first = min_heap[0]

            for i in range(groupSize):
                card = first + i
                if count[card] == 0:
                    return False

                count[card] -= 1
                if count[card] == 0:
                    if card != heapq.heappop(min_heap):
                        return False
        
        return True
```

### Key Concepts:
- We use a Hashmap to track the number of each card we have left
- We use a min heap to track the minimum card available
- While heap, we want to build a group
- We grab the card at the top of the heap, and we try to build a group with it. 
- Each card we decrement the counter, and then we check if its reached 0
- If the card count has reached 0, we check if it was the smallest card in the heap
- If not, that means we created a "hole" and can't create a valid group

### Time and Space:
- Time: O(n log n), where n is the number of unique cards
    - Sorting takes O(n log n)
    - Processing each card takes O(groupSize) which is constant
- Space: O(n) for the counter dictionary

## Merge Triplets to Form Target
You are given a 2D array of integers triplets, where triplets[i] = [ai, bi, ci] represents the ith triplet. You are also given an array of integers target = [x, y, z] which is the triplet we want to obtain.

To obtain target, you may apply the following operation on triplets zero or more times:

Choose two different triplets triplets[i] and triplets[j] and update triplets[j] to become [max(ai, aj), max(bi, bj), max(ci, cj)].
* E.g. if triplets[i] = [1, 3, 1] and triplets[j] = [2, 1, 2], triplets[j] will be updated to [max(1, 2), max(3, 1), max(1, 2)] = [2, 3, 2].

Return true if it is possible to obtain target as an element of triplets, or false otherwise.

```python
# return true if it is possible to obtain target
# Greedy approach: Check if triplet is valid, check each position in valid triplet, 
# and if it matches the same target position, add that index to the good set
# return true if the good set is length 3 (we found a match for all 3 indices)
class Solution:
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        good = set()

        for triplet in triplets:
            # Check if valid triplet
            if triplet[0] <= target[0] and triplet[1] <= target[1] and triplet[2] <= target[2]:

                for i in range(3):
                    if triplet[i] == target[i]:
                        good.add(i)
        
        return len(good) == 3
```

### Key Concepts:
- We can find valid triplets by checking if each position in the triplet is less than or equal to the same position in target. 
- Once we find a valid triplet, we want to check each position in the triplet, and see if any of them match the corresponding position in target
- If we find a match, we add that index to the good set, it means that we have a valid triplet that can be merged to satisfy that position

### Time and Space Complexity
- Time: O(n)
- Space: O(1)