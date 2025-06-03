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