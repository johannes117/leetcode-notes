## Subsets
Given an array nums of unique integers, return all possible subsets of nums.

The solution set must not contain duplicate subsets. You may return the solution in any order.

```python
# Use Recursive Backtracking to either pick the current number or not pick it
# initialise n, res and sol global variables. 
# define backtracking helper function with index input
# basecase: if index is equal to n (len of nums), we want to append a copy of the solution list to the result list. using this syntax: sol[:]. Return
# call backtrack on the don't pick (go "left")
# append current character to sol list, then call backtrack (go right)
# pop

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        result, solution = [], []

        def backtrack(index):
            if index == n:
                result.append(solution[:])
                return
            
            # Go Left / Don't Pick
            backtrack(index + 1)
            
            # Go Right / Pick
            solution.append(nums[index])
            backtrack(index + 1)
            solution.pop()
        
        backtrack(0)
        return result
```

### Key Concepts:
- To find every permutation of the input list, we need to use recursive backtracking. 
- We use DFS to implement the backtracking algorithm
- Once we find the basecase, we "backtrack" and undo that change. Thats what the pop is for. 
- Other than that its basically just DFS

### Time Complexity
- Time: O(2^n)
- Space: O(n), due to the recursive callstack

## Combination Sum
You are given an array of distinct integers nums and a target integer target. Your task is to return a list of all unique combinations of nums where the chosen numbers sum to target.

The same number may be chosen from nums an unlimited number of times. Two combinations are the same if the frequency of each of the chosen numbers is the same, otherwise they are different.

You may return the combinations in any order and the order of the numbers in each combination can be in any order.

```python
# Basecases: current sum is target (append), or current sum is greater than target, or index = len of candidates. 
# Go Right: Don't use number anymore, backtrack with index + 1 and current sum
# append
# Go Left: Use number, backtrack with index and current sum + candidate[index]
# pop

class Solution:
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        res, sol = [], []
        n = len(nums)

        def backtrack(index, cur_sum):
            if cur_sum == target: # found a solution
                res.append(sol[:])
                return
            
            if cur_sum > target or index == n:
                return # Did not find a solution

            # Don't pick the current number, "move on"
            backtrack(index + 1, cur_sum) # current sum stays the same because we aren't adding the current number
            sol.append(nums[index])
            backtrack(index, cur_sum+nums[index])
            sol.pop()
        
        backtrack(0,0)
        return res
```

### Key Concepts:
- Very similar Recursive Backtracking Formula: Basecase, a solution and a result global variable. 
- Basecases: current sum is equal to the target (found an answer), current sum greater than target (not an answer), index equals length of nums (reached the last permutation)
- Going Right: We "don't" use the current number at this index anymore, so we increment the index, and pass in the current sum
- Going Left: We "do" use the current number, its appended to the sol list, and we add it to the current sum
- After going left we need to remove the current number from the solution list, so we pop

### Time and Space Complexity:
- Time: approximately O(n**t)
- Space: O(n)