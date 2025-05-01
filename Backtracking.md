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

## Subsets II

You are given an array nums of integers, which may contain duplicates. Return all possible subsets.

The solution must not contain duplicate subsets. You may return the solution in any order.

```python
# return all possible subsets from the input array nums. Contains duplicates.
# Sort the input array
# Base case reached the ens of the array
# Include the current element (append then pop)
# Skip all duplicates at this level (while loop)
# exclude the current element and its duplicates.

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res, sol = [], []
        nums.sort()

        def backtrack(i, sol):
            # Basecase: reached the end of the array
            if i == len(nums):
                res.append(sol.copy())
                return

            # Include the current element
            sol.append(nums[i])
            backtrack(i + 1, sol)
            sol.pop() # backtrack

            # skip duplicates
            while i + 1 < len(nums) and nums[i] == nums[i+1]:
                i += 1

            # Don't include current element
            backtrack(i+1, sol)

        backtrack(0, sol)
        return res
```

### Key Concepts

- Similar solution to the Combination2 solution
- We can handle duplicates using a while loop that will increment the pointer in a sorted array until we have reached the last duplicate for that element.
- Basecase: reached the end of the array
- Decision 1: Include Element then backtrack
- Skip Duplicates:
- Decision 2: Don't include element

## Word Search

Given a 2-D grid of characters board and a string word, return true if the word is present in the grid, otherwise return false.

For the word to be present it must be possible to form it with a path in the board with horizontally or vertically neighboring cells. The same cell may not be used more than once in a word.

```python
# m x n grid of characters board, and a string word
# return true if word exists in grid.
# check for empty inputs
# initialise rows and cols length variables
# Helper function for DFS: row column and word index.
# Base case: if we've matched all characters in the word (index equals length of word) return true
# out of bounds or character doesn't match return false
# Mark the cell as visited by changing to # (store in temp)
# explore all 4 directions.
# restore the cell (backtrack)
# return found
# try starting from each cell in the grid.
# nested for loop rows and cols
# if character equals first c in word, and dfs returns true return True
# otherwise return false.
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board or not word:
            return False

        rows, cols = len(board), len(board[0])

        def dfs(r, c, index):
            # Base case: if index is len word
            if index == len(word):
                return True

            # Check for out of bounds or if character doesn't match
            if r < 0 or r >= rows or c < 0 or c >= cols or word[index] != board[r][c]:
                return False

            # replace character with #
            temp = board[r][c]
            board[r][c] = "#"

            # check every direction:
            found = (
                dfs(r+1, c, index+1) or
                dfs(r-1, c, index+1) or
                dfs(r, c+1, index+1) or
                dfs(r, c-1, index+1)
            )

            #backtrack, put character back
            board[r][c] = temp
            return found

        # Try every element in the grid to start
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == word[0] and dfs(r, c, 0):
                    return True

        return False
```

### Key Concepts

- The basecase is if the index reaches the length of the word
- We use a nested for loop to start with every element in the grid
- if we reach an element that matches the first letter in the word, we want to call our dfs function on it
- DFS function takes in the row, the column and the current index of the word that we are checking.
- We need to handle out of bounds checks, and also whether the character we are visiting is not what we are looking for.
- We use a temporary variable to store the current character, and then we replace it with a special character "#"
- This is so that we can mark that cell in the grid as visited, we will "backtrack" and undo this later if needed
- We want to check every direction, which means we need to call dfs with row +-1 and col +-1 to check for the next letter in the word (index+1)
- if any direction returns true, the found variable will be true.
- after checking every direction from the current element we want to backtrack and restore from the temporary variable.
- we can simply return the found variable at the end of the helper function.

### Time and Space Complexity

- Time: O(N x M x 4^L), where n and m represent the rows and columns and L represents the length of the word.
- Space: O(L), where l is the length of the word due to the Recursion stack. We are using in place modification so no additional space for board.



## Palindrome Partitioning
Given a string s, split s into substrings where every substring is a palindrome. Return all possible lists of palindromic substrings.

You may return the solution in any order.

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res, sol = [], []

        def backtrack(i, sol):
            # Base case: reached the end of the string
            if i >= len(s):
                res.append(sol.copy())
                return 
            
            # try all possible substrings
            for j in range(i+1, len(s)+1):
                substring = s[i:j]
                if substring == substring[::-1]:
                    sol.append(substring)
                    backtrack(j, sol)
                    sol.pop() # backtrack

        backtrack(0, sol)
        return res
```

### Key Concepts:
- We are trying to find different partitions instead of permutations. Think of it as slicing the string in place to create substrings. 
- We want to return a list of substrings that are palindromes (same backwards)
- Basecase: our index has reached the end of the string. We want to append our current path to the result list

### Time and Space Complexity
- Time: O(n*2^n), where n is the length of the string. This is because there are 2^N number of ways to partition a string, 
and each partition we have to use O(n) time to check if the substrings are palindromes
- Space: O(n), due to the height of the recursive callstack

## Letter Combinations of a Phone Number
You are given a string digits made up of digits from 2 through 9 inclusive.

Each digit (not including 1) is mapped to a set of characters as shown below:

A digit could represent any one of the characters it maps to.

Return all possible letter combinations that digits could represent. You may return the answer in any order.

```python
# edge case empty list
# build phone map
# result and sol lists
# backtrack function with index and sol
# basecase is index len of digits. use ''.join()
# get all possible letters for the current digit
# try each letter and recurse
# add letter to sol
# call backtrack
# pop
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return []
        
        phone_map = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }

        res, sol = [], []

        def backtrack(index, sol):
            # Basecase
            if index == len(digits):
                res.append(''.join(sol))
                return
            
            letters = phone_map[digits[index]]

            for letter in letters:
                sol.append(letter)
                backtrack(index+1, sol)
                sol.pop()

        
        backtrack(0, sol)
        return res
```

### Key Concepts:
- Backtracking with a hashmap 

### Time and Space Complexity
- Time: O(4^n), because each n multiplies by at most 4
- Space: O(n), height of the callstack