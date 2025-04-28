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

- Time: approximately O(n\*\*t)
- Space: O(n)

## Combination Sum II

You are given an array of integers candidates, which may contain duplicates, and a target integer target. Your task is to return a list of all unique combinations of candidates where the chosen numbers sum to target.

Each element from candidates may be chosen at most once within a combination. The solution set must not contain duplicate combinations.

You may return the combinations in any order and the order of the numbers in each combination can be in any order.

```python
# Res and Sol lists
# Sort the list
# dfs backtrack function with index, current and total
# basecases:
# - total equals target (found),
# - total greater than target or index equal len of candidates
# recursive cases: include candidate at index or skip
# include, append candidate, run dfs, pop candidate
# skip, bump index while candidate equals index + 1
# run dfs
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res, sol = [], []
        candidates.sort()

        def backtrack(i, sol, total):
            # Found Basecase:
            if total == target:
                res.append(sol.copy())
                return
            if total > target or i == len(candidates):
                return

            # include candidate
            sol.append(candidates[i])
            backtrack(i + 1, sol, total+candidates[i])
            sol.pop()
            # skip candidate
            while i + 1 < len(candidates) and candidates[i] == candidates[i+1]:
                i += 1
            backtrack(i + 1, sol, total)

        backtrack(0, sol, 0)
        return res
```

### Key Concepts

- Similar solution to combination sum 1, instead we sort the candidates, and use a while loop to skip duplicates.

### Time and Space Complexity

- Time: approximately O(n\*\*t)
- Space: O(n)

## Permutations

Given an array nums of unique integers, return all the possible permutations. You may return the answer in any order.

```python
# problem: array of unique integers. Return all possible permutations.
# res and sol lists for backtracking
# basecase: if the length of solution is equal to n. Append copy of sol to res.
# loop through nums, of x not in sol, append x and call backtrack, then pop.

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res, sol = [], []
        n = len(nums)

        def backtrack():
            if len(sol) == n:
                res.append(sol.copy())
                return

            for x in nums:
                if x not in sol:
                    sol.append(x)
                    backtrack()
                    sol.pop()

        backtrack()
        return res
```

### Key Concepts

- Slight variation on DFS solution
- Basecase is if the length of the solution list is equal to the length of the input array. We append a copy of that solution to the result list
- Each time we call backtrack, we want to loop through every element in the input array that is not already in the solution list.
- Every time we add a number, we call backtrack again, so we can exhaust all possible permutations along that path before backtracking (popping)
- This for loop inside the backtrack function allows us to check all permutations.

### Time and Space Complexity

- Time: O(n!), each level is n times larger than the level before
- Space: approx O(n), where n is the height of the callstack.
