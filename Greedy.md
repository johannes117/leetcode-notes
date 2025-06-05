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