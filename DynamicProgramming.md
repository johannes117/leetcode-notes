# Dynamic Programming

## Fibonacci Number
The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,

F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.
Given n, calculate F(n).

```python
# Use a memo (dictionary) to remember previously computed functions. Initialise with our basecases f(0) = 0, f(1) = 1
# helper function f(x), check if x in memo, return value, else compute x and store in memo
# return f(n)
class Solution:
    def fib(self, n: int) -> int:
        memo = {0:0, 1:1}

        def f(x):
            if x in memo:
                return memo[x]
            else:
                memo[x] = f(x-1) + f(x-2)
                return memo[x]
        
        return f(n)
```

### Key Concepts:
- Top down DP using Memoization

### Time and Space:
- Time: O(n)
- Space: O(n)

## Climbing Stairs
You are given an integer n representing the number of steps to reach the top of a staircase. You can climb with either 1 or 2 steps at a time.

Return the number of distinct ways to climb to the top of the staircase.

### Top Down Dynamic Programming solution
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {1:1, 2:2}

        def f(x):
            if x in memo:
                return memo[x]
            else:
                memo[x] = f(x-1) + f(x-2)
                return memo[x]
        
        return f(n)
```

### Key Concepts:
- Top Down Memoization uses a memo or "cache" to store the solutions to subproblems that we have already computed but might need later. 

### Time and Space
- Time: O(n)
- Space: O(n)

### Bottom Up Dynamic Programming Solution (Tabulation)
```python
# Bottom up tabulation
# define edgecases: if n is 1 return 1, if 2 return 2
# create a list of 0s of length n
# set first two positions to 1 and 2 respectively (these are the base cases)
# loop from 2 to n, set the value at the current index to the sum of the previous 2 indexes
# return the last value in the list. 
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2

        dp = [0] * n
        dp[0] = 1
        dp[1] = 2 

        for i in range(2, n):
            dp[i] = dp[i-2] + dp[i-1]
        
        return dp[n-1]
```

### Key Concepts:
- We start at the first case and work our way up, creating a table of results. 
- Our answer will be the last result in the list/table. 

### Time and Space
- Time: O(n)
- Space: O(n)

### Bottom Up Constant Space

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2

        prev = 1
        curr = 2 

        for i in range(2, n):
            prev, curr = curr, prev+curr
        
        return curr
```

### Key Concepts:
- We don't actually need to store the results of each step in the table because all we want it the final answer
- So we can simply use two pointers to keep track of the current and previous positions
- For each loop we set the current pointer to the sum of both pointers, and the prev pointer to the current pointer and so forth
- We just return the curr pointer at the end which will contain our result for n steps. 

### Time and Space:
- Time: O(n)
- SpacE: O(1)

## Min Cost Climbing Stairs
You are given an array of integers cost where cost[i] is the cost of taking a step from the ith floor of a staircase. After paying the cost, you can step to either the (i + 1)th floor or the (i + 2)th floor.

You may choose to start at the index 0 or the index 1 floor.

Return the minimum cost to reach the top of the staircase, i.e. just past the last index in cost.

```python
# given a cost array, return the minimum cost to reach the top of the staircase
# Bottom Up DP Constant
# intialise n, prev and curr
# loop from index 2 to n+1
# setup prev to curr, and set curr to the minimum of cost 2 steps back plus prev, or cost 1 step back of curr. 
#return curr
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        prev, curr = 0, 0

        for i in range(2, n+1):
            prev, curr = curr, min(cost[i-2] + prev, cost[i-1] + curr)
        
        return curr
```

### Key Concepts
- Similar to Fibonacchi and Climbing stairs: We can solve it using Top down memoized, or bottom up tabulization or bottom up constant.
- For constant we just keep a prev and curr pointer, which tracks what the cost of the previous two steps were. 
- When we move to the next set of steps, set prev to curr, and set curr to the mimimum of the cost 2 steps back + prev, and the cost 1 step back + curr. 
- return curr

### Time and Space:
- Time: O(n)
- Space: O(1)