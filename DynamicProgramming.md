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


## House Robber
You are given an integer array nums where nums[i] represents the amount of money the ith house has. The houses are arranged in a straight line, i.e. the ith house is the neighbor of the (i-1)th and (i+1)th house.

You are planning to rob money from the houses, but you cannot rob two adjacent houses because the security system will automatically alert the police if two adjacent houses were both broken into.

Return the maximum amount of money you can rob without alerting the police.

```python
# return maximum ammount of money you can rob without alerting the police
# trick: maximum amount of any given house is the max of not robbing it or robbing it. 
# Robbing the House: the current value of the house + the value of the house 2 steps back 
# Not Robbing the House: just the current value of the house. 
# base cases: n == 1: nums[0], n ==2: max(nums[0], nums[1])
# prev and curr pointers initialised to base cases
# loop from 2 to n
# set prev to curr, and curr to the max of not robbing or robbing the house
# return curr
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1: return nums[0]
        if n == 2: return max(nums[0], nums[1])

        prev, curr = nums[0], max(nums[0], nums[1])

        for i in range(2, n):
            prev, curr = curr, max(prev + nums[i], curr)
        
        return curr
```

### Key Concepts:
- Very similar to Climbing Stairs and fibonacci but with a trick
- We either want to rob a house, or not rob it.
- robbing the house, means we want to take the value of the house 2 steps back and add it to the value of the current house. 
- not robbing the house, means that we just take the value of the house 1 step back. 
- We take the max of robbing or not robbing the house, and that becomes the new curr pointer. 
- This is a bottom up approach, because we are starting from the bottom, base case and iteratively moving through the array and keeping track of the totals as we go. 
- This constant space optimisation doesn't use tabulation since we only care about the last two houses. 

### Time and Space:
- Time: O(n)
- Space: O(1)

## House Robber 2
You are given an integer array nums where nums[i] represents the amount of money the ith house has. The houses are arranged in a circle, i.e. the first house and the last house are neighbors.

You are planning to rob money from the houses, but you cannot rob two adjacent houses because the security system will automatically alert the police if two adjacent houses were both broken into.

Return the maximum amount of money you can rob without alerting the police.

```python
class Solution:
    def rob(self, nums: List[int]) -> int:     
        if len(nums) == 0: return 0
        if len(nums) == 1: return nums[0]
        if len(nums) == 2: return max(nums[0], nums[1])
        def helper(nums):
            prev, curr = nums[0], max(nums[0], nums[1])
            for i in range(2, len(nums)):
                prev, curr = curr, max(curr, prev + nums[i])
            return curr
        return max(helper(nums[1:]), helper(nums[:-1]))
```

### Key Concepts:
- Same as house robber 1, 
- just slice the array and perform the algorithm on all of the nums except the first, and all of the nums except the last

### Time and Space:
- Time: O(n)
- Space: O(1)

## Longest Palindromic Substring
Given a string s, return the longest substring of s that is a palindrome.

A palindrome is a string that reads the same forward and backward.

If there are multiple palindromic substrings that have the same length, return any one of them.

```python
# Expand around the center approach
# Edgecase: if not s return ""
# initialise start variable: starting index of the longest palindromic substring
# initialise max_length to 1: length of the longest palindromic substring
# helper function to expand around center: left and right pointers as arguments
# while left and right are inbounds, and the characters at the left and right indices are equal: increment pointers
# return right - left - 1: The length of palindrome. 
# loop for length of s
# call helper function twice for odd and even, store in len1 and len2
# Find the maximum length from both expansions
# Update if we found a longer palindrome: if length is greater than max_length, update it, and calculate the new starting index of the substring
# start = i - (length - 1) // 2
# return substring from start to start + max_length
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s: return ""
        start = 0
        max_len = 1

        def expand_around_center(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        for i in range(len(s)):
            len1 = expand_around_center(i, i)
            len2 = expand_around_center(i, i+1)

            length = max(len1, len2)

            if length > max_len:
                max_len = length
                start = i - (length - 1 ) // 2
        
        return s[start:start+max_len]
```

### Key Concepts:
- We can solve using an "Expand around the center" approach
- Essentially take a left and a right pointer and shift them left and right aslong as the pointers are in bounds and equal to eachother. 
- Once these conditions are not met, we have found the longest palindromic substring for that index, and we return the length of that substring
- We call the helper function twice, once for odd and once for even for every index. Odd: i, i and Even: i, i + 1
- Take the maximum of the two calculations, and if its greater than our global maximum we update our global variables. 
- For the start index, we need to calculate it using the max length: i - (length - 1) // 2
- We simply return the string from the start index to the start index + the max length calculated

### Time and Space:
- O(n^2): Worst case we visit every index, and have to expand for every character in the string. 
- O(1)

## Palindromic Substrings
Given a string s, return the number of substrings within s that are palindromes.

A palindrome is a string that reads the same forward and backward.

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        if not s: return 0
        count = [0]

        def helper(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
                count[0] += 1
        
        for i in range(len(s)):
            helper(i, i)
            helper(i, i+1)
        
        return count[0]
```

### Key Concepts:
- Almost exactly the same as the Longest palindromic substring, although this time we simply increment the count for every found substring. 
- For a given starting index for left and right pointers, while pointers are inbound, and the values at the pointers are equal, we have found a palndromic substring. Increment counter
- Call helper function for odd and even: i, i and i, i + 1 respectively
- Return count. 

### Time and Space:
- Time: O(n^2), we have to iterate through every character in the string, and may have to loop through every substring of length n in the worst case. 
- Space: O(1)


## Decode Ways
A string consisting of uppercase english characters can be encoded to a number using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
To decode a message, digits must be grouped and then mapped back into letters using the reverse of the mapping above. There may be multiple ways to decode a message. For example, "1012" can be mapped into:

"JAB" with the grouping (10 1 2)
"JL" with the grouping (10 12)
The grouping (1 01 2) is invalid because 01 cannot be mapped into a letter since it contains a leading zero.

Given a string s containing only digits, return the number of ways to decode it. You can assume that the answer fits in a 32-bit integer.

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0': return 0
        n = len(s)
        dp = [0] * (n+1)
        dp[0], dp[1] = 1, 1

        for i in range(2, n+1):
            # Single Digit Decode
            if s[i-1] != '0':
                dp[i] += dp[i-1]

            # Double Digit Decode
            two_digit = int(s[i-2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i-2]
        
        return dp[-1]
```

### Key Concepts:
- Can be solved using Bottom Up Tabulation (or Bottom Up Constant if we use the pointers)
- Edgecases: not s or first character is 0 then we return 0 immediately. 
- Basecases: The first two positions in the DP array are both set to 1 as base cases. 
- Explanation: dp[0] in this case represents an empty string, which can only be decoded 1 way. 
- dp[1] has already been checked as non 0. no matter what number this is, it will only have 1 way to be decoded. 
- Hence our base cases are positions 0 and 1 as 1 and 1 respectively. 
- We then loop from 2 to n + 1 (the end of the dp array)
- For each index, we want to check its single decode and double decode.
- For single, aslong as the previous character was not 0, we add the previous value to our current value
- For double, we use sliding to convert the indexes into an integer. We then check if its between 10 and 26.
- If so then its a valid double digit, and can be decoded, so we add it to the tally for that index in the dp array. 
- We return the final value in the dp table, as it will contain the accumulated number of decodings. 
- We can optimmise the space by using pointers since we only care about the previous two indexes. 

### Time and Space
- Time: O(n)
- Space: O(n) (Can be optimised)


## Coin Change
You are given an integer array coins representing coins of different denominations (e.g. 1 dollar, 5 dollars, etc) and an integer amount representing a target amount of money.

Return the fewest number of coins that you need to make up the exact target amount. If it is impossible to make up the amount, return -1.

You may assume that you have an unlimited number of each coin.

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort()
        dp = [0] * (amount + 1)

        for i in range(1, amount+1):
            minimum = float('inf')

            for coin in coins:
                diff = i - coin
                if diff < 0:
                    break
                minimum = min(minimum, dp[diff] + 1)
            
            dp[i] = minimum
        
        if dp[amount] < float('inf'):
            return dp[amount]
        else:
            return -1
```

### Key Concepts:
- Can be solved using Top Down Memoization or Bottom Up Tabulation
- Bottom Up approach: Maintain a dp array of pre-calculated values that we can use to calculate later values using their diffs with the target amount. 
- Since we track the minimum number of coins to add up to each index amount in the dp array, we can simply refer to previous values in the dp array to calculate a new minimum amount for each coin
- We update the minimum value if its the smallest amount for that iteration
- Once we have checked all the coins, we set the dp[index] to the minimum we found. 
- Once the loop completes we should have computed the minimum number of coins needed for the target amount which is the last element in the dp array. 

### Time and Space
- Time: O(Coins * Amount)
- Space: O(Amount)

## Maximum Product Subarray
Given an integer array nums, find a subarray that has the largest product within the array and return it.

A subarray is a contiguous non-empty sequence of elements within an array.

You can assume the output will fit into a 32-bit integer.

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums: return 0

        max_so_far = nums[0]
        min_so_far = nums[0]
        result = nums[0]

        for i in range(1, len(nums)):
            curr = nums[i]

            # calculate new min and max
            temp = max(curr, max_so_far * curr, min_so_far * curr)
            min_so_far = min(curr, max_so_far * curr, min_so_far * curr)
            max_so_far = temp

            result = max(result, max_so_far)
        
        return result
```

### Key Concepts:
- We can solve this by maintaining a min so far and a max so far, and when we calculate the max we have three outcomes:
-  Curr: start a new subarray here
- max_so_far * curr: Extend the best subarray from previous position
- min_so_far * curr: Extend the worst subarray from previous position (might become best if curr is negative)
- check if not nums, return 0
- initialise max_so_far, min_so_far, result to the first element in nums
- loop from 1 to len nums
- set curr to the current nums value at index
- Calculate new max and min of curr, max_so_far * curr, min_so_far * curr
- update the global maximum
- return result

### Time and Space:
- Time: O(n)
- Space: O(1)

## Word Break
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of dictionary words.

You are allowed to reuse words in the dictionary an unlimited number of times. You may assume all dictionary words are unique.

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # Convert to set for O(1) lookup
        word_set = set(wordDict)
        n = len(s)
        # dp[i] represents if s[0:i] can be segmented
        dp = [False] * (n+1)
        dp[0] = True # Basecase

        # Check each position
        for i in range(1, n + 1):
            # Try all possible previous positions
            for j in range(i):
                # If we can segment up to j and s[j:i] is a valid word
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break # Found one valid segmentation, no need to continue
        
        return dp[n]
```

### Key Concepts:
- We can use Bottom Up DP to solve this problem.
- We initialise a dp array of False values of length string + 1 (We need to include the basecase of length 0)
- Basecase: dp[0] = True because an empty can always be segmented
- we loop from 1 to n + 1 (basecase at 0 has already been set)
- try all previous positions basically loop from 0 to if
- if the j position in the dp array is True, then we check if the remaining substring is in our wordDict
- Since basecase at index 0 is true, eventually we will find a substring from 0 to the first split point
- Example "leetcode", at position 4 dp = [T, F, F, F, T, ...] we have detected that we were able to split "leet" because leet is in our word dictionary
- and "" is a valid segment. 
- When we reach position 8, and j reaches position 4, we are then left with s[j:i] = "code" which is in our word set
- we would then set dp[8] to true, which will be our final answer!

### Time and Space
- Time: O(n^2 * m): We have a nested for loop for O(n^2) and we have a string lookup which is O(m) where m is the length of the word
- Space: O(n + k), where n is length of string and k is the length of the word dict. 


## Longest Increasing Subsequence
Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from the given sequence by deleting some or no elements without changing the relative order of the remaining characters.

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
```

### Key Concepts:
- Almost identical algorithm to Word Break. We can use Bottom up DP (Tabulation) to store a maximum increasing subsequence at every index. 
- Basecase: dp[0] = 1, the first number is itself a subsequence and since its the first it will automatically be 1. 
- For every position i, we check each previous position, if its smaller, we want to set our current longest subsequence to the max of our current longest vs the max of the previous position + 1 (ourself)
- We return the max value in dp since the answer won't necessarily be the last one in the dp table. 

### Time and Space:
- Time: O(n^2), due to the nested for loop
- Space: O(n), due to the dp array of length n

## Partition Equal Subset Sum
You are given an array of positive integers nums.

Return true if you can partition the array into two subsets, subset1 and subset2 where sum(subset1) == sum(subset2). Otherwise, return false.

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)

        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2

        dp = [False] * (target + 1)
        dp[0] = True

        for num in nums:
            for j in range(target, num -1, -1):
                dp[j] = dp[j] or dp[j-num]
        
        return dp[target]
```

### Key Concepts:
- This problem can be rewritten as: Can you find a subset that can sum up to the total sum // 2?
- We can use bottom up DP (Tabulation) to build up a DP array where each index determines whether we can sum up to that index using the numbers available. 
- The basecase is dp[0] = True, because 0 can always be summed up to. 
- each iteration, we want to update the value at dp[j] when j - num (the current number in the iteration) is an integer that we previously computed as summable. 
- In each inner for loop, we iterate backwards, this prevents using the same number twice. 

### Time and Space
- Time: O(n * sum)
- Space: O(sum)

## Unique Paths
There is an m x n grid where you are allowed to move either down or to the right at any point in time.

Given the two integers m and n, return the number of possible unique paths that can be taken from the top-left corner of the grid (grid[0][0]) to the bottom-right corner (grid[m - 1][n - 1]).

You may assume the output will fit in a 32-bit integer.

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = []
        for _ in range(m):
            dp.append([0]*n)

        dp[0][0] = 1

        for i in range(m):
            for j in range(n):
                if i == j == 0:
                    continue
                val = 0
                if i > 0:
                    val += dp[i-1][j]
                if j > 0:
                    val += dp[i][j-1]
                dp[i][j] = val
        
        return dp[m-1][n-1]
```

### Key Concepts:
- This first row and the first column, only ever have 1 way to reach those cells
- Out of bounds can be considered as a '0', not adding anything. 
- We can solve this using Bottom Up DP (Tabulation)
- Each position in the dp array represents the number of ways you can reach that position from the starting position.
- We iterate through the grid normally using a nested for loop i, j. 
- We first check to make sure we are not at the basecase/starting position
- We initialise a val variable which we will use to accumulate the number of ways we can reach our current position
- we check if i > 0, this means that theres a value above us (means that we are on atleast the second row). If so we want to actually add the value of the position above us to our val
- We do the same thing with j, we check if theres a value to our left, if so we add it to our tally. 
- We then set our current position to our computed Val total
- Once we reach the final position we will be left with the total number of ways to reach that position. 

### Time and Space
- Time: O(m*n)
- Space: O(m*n)

## Longest Common Subsequence
Given two strings text1 and text2, return the length of the longest common subsequence between the two strings if one exists, otherwise return 0.

A subsequence is a sequence that can be derived from the given sequence by deleting some or no elements without changing the relative order of the remaining characters.

For example, "cat" is a subsequence of "crabt".
A common subsequence of two strings is a subsequence that exists in both strings.

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)

        memo = {}
        def longest(i, j):
            if (i, j) in memo:
                return memo[(i,j)]
            if i == m or j == n:
                memo[(i, j)] = 0
                return memo[(i, j)]
            elif text1[i] == text2[j]:
                memo[(i, j)] = 1 + longest(i+1, j+1)
                return memo[(i, j)]
            else:
                memo[(i,j)] = max(longest(i, j+1), longest(i+1, j))
                return memo[(i,j)]
        
        return longest(0,0)
```

### Key Concepts
- We essentially use a pointer to iterate through each string
- If both pointers are equal, increment both
- else, we want to get the max of incrementing one of them and not the other. (Decision Tree)
- We can use Top down DP using Memoization to solve this problem
- Initialise m and n
- initialise helper function with i and j
- check if in bounds, return 0
- check if match, return 1 + helper i and j incremented
- if no match, return max of moving j and moving i
- return helper at 0, 0

### Time and Space:
- Time: O(m*n)
- Space: O(m*n)

## Best Time to Buy and Sell Stock with Cooldown
You are given an integer array prices where prices[i] is the price of NeetCoin on the ith day.

You may buy and sell one NeetCoin multiple times with the following restrictions:

After you sell your NeetCoin, you cannot buy another one on the next day (i.e., there is a cooldown period of one day).
You may only own at most one NeetCoin at a time.
You may complete as many transactions as you like.

Return the maximum profit you can achieve.

```python
# return the maximum profit you can achieve
# This can be solved using Bottom Up DP (Tabulation) Space Optimised
# Since we only need the previous days values, we can compress the table into three variables: hold, sell, rest
# hold: max profit when holding a stock
# sold: max profit when just sold (cooldown day)
# rest: max profit when resting (can buy)
# chack if prices is less than or equal to 1, return 0
# initialise hold, sold, rest as -prices[0], 0 , 0
# loop from 1 to prices
# create prev_ variables 
# To hold today: either keep holding or but today (from rest state)
# To sell today: sell the stock we are holding
# To rest today: either keep resting or finish cooldown
# at the end, we want to either be resting or have just sold.  
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0

        hold = -prices[0] # Buy first stock
        sold = 0 # Can't sell first stock
        rest = 0

        for i in range(1, len(prices)):
            prev_hold = hold
            prev_sold = sold
            prev_rest = rest

            # Buy or Hold
            hold = max(prev_hold, prev_rest - prices[i])

            # To Sell Today
            sold = prev_hold + prices[i]

            # To rest today 
            rest = max(prev_rest, prev_sold)
        
        return max(sold, rest)
```

### Key Concepts
- Bottom Up DP Space Optimised is used in this solution. 
- We essentially track the decision made in the previous iteration
- Decision to buy, we take the max of the previous hold, or previous rest - prices[i]
- Decision to Sell, we take the previous hold and add price at current index
- Decision to Rest, we take the max of previos rest or previous sold
- We return the maximum between the sold and rest values

### Time and Space:
- Time: O(n)
- Space: O(1)

## Coin Change 2
You are given an integer array coins representing coins of different denominations (e.g. 1 dollar, 5 dollars, etc) and an integer amount representing a target amount of money.

Return the number of distinct combinations that total up to amount. If it's impossible to make up the amount, return 0.

You may assume that you have an unlimited number of each coin and that each value in coins is unique.

```python
# return the number of ways we can make up the amount using the available coins
# Bottom Up DP (Tabulation)
# Create a DP array where dp[i] = number of ways to make amount i
# basecase: dp[0] = 1 (one way to make 0, use no coins)
# For each coin, update all amounts that can be formed using that coin
# for each amount, add the number of ways from amount - coin
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1

        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]

        return dp[-1]
```

### Key Concepts
Algorithm
1. Create a DP array where dp[i] = number of ways to make amount i
2. Initialize dp[0] = 1 (one way to make amount 0: use no coins)
3. For each coin, update all amounts that can be formed using that coin
4. For each amount, add the number of ways from amount - coin
- Order matters in iteration: We iterate through coins first, then amounts. This prevents counting the same combination multiple times (e.g., [1,2] and [2,1] as different ways).
- Building up solutions: For each amount, we consider adding each coin and look up how many ways we can make the remaining amount.

### Time and Space:
- Time: O(m*n)
- Space: O(n)

## Target Sum
You are given an array of integers nums and an integer target.

For each number in the array, you can choose to either add or subtract it to a total sum.

For example, if nums = [1, 2], one possible sum would be "+1-2=-1".
If nums=[1,1], there are two different ways to sum the input numbers to get a sum of 0: "+1-1" and "-1+1".

Return the number of different ways that you can build the expression such that the total sum equals target.

```python
# Return the number of ways we can sum up to target with our nums list
# Top Down Memoization: recursive helper with index, and curr sum
# check if in memo
# if we have reached the len of nums, return 1 if current sum is target else 0
# we want to store the value returned of both subtracting or adding the current number to the sum at the current position in the memo
# return current position in memo
# call helper function at position 0, sum 0
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        memo = {}
        n = len(nums)

        def dfs(index, curr_sum):
            if (index, curr_sum) in memo:
                return memo[(index, curr_sum)]
            
            if index == n:
                return 1 if curr_sum == target else 0
            
            memo[(index, curr_sum)] = (
                dfs(index + 1, (curr_sum + nums[index])) +
                dfs(index + 1, (curr_sum - nums[index]))
            )
            return memo[(index, curr_sum)]

        return dfs(0,0)
```

### Key Concepts (Top Down)
- Standard Top Down approach
- We can either add or subtract a number at each index to add to the target sum.
- We do a DFS until we hit our basecase of index == n, this means we have hit a leaf node
- If that leaf node is equal to our target, we want to count that as a way to sum up to that target
- If not, then we return 0
- That means our helper function will return the number of ways that branch could sum up to the target
- We add simple memoization to prevent recalculating previous values and to improve the time complexity. 

### Time and Space
- Time: O(m*n)
- Space: O(m*n)

Note: This can be improved with Bottom Up Space optimised. 

## Interleaving String
You are given three strings s1, s2, and s3. Return true if s3 is formed by interleaving s1 and s2 together or false otherwise.

Interleaving two strings s and t is done by dividing s and t into n and m substrings respectively, where the following conditions are met

|n - m| <= 1, i.e. the difference between the number of substrings of s and t is at most 1.
s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
Interleaving s and t is s1 + t1 + s2 + t2 + ... or t1 + s1 + t2 + s2 + ...
You may assume that s1, s2 and s3 consist of lowercase English letters.

```python
# Return true if s3 is formed by interleaving s1 and s2
# Top Down DP Memoization approach
# Edgecase: check if s1 and s2 add up to s3
# initialise memo
# DFS helper function with 3 indexes: i, j, k
# Basecase: reached end of all strings
# Basecase 2: out of bounds in s3 (k)
# check in memo
# initialise result variable to false
# Try taking a character from s1: if i inbounds and character equals s3 at k
# set result to result or the dfs of i + 1, j and k + 1 (increment i and k)
# do the same for taking a character from s2
# store result in memo then return it
# return dfs(0,0,0_)
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        
        memo = {}

        def dfs(i, j, k):
            if i == len(s1) and j == len(s2) and k == len(s3):
                return True
            
            # k out of bounds
            if k >= len(s3):
                return False

            # check memo
            if (i,j) in memo:
                return memo[(i,j)]

            result = False

            # Try character from first string
            if i < len(s1) and s1[i] == s3[k]:
                result = result or dfs(i + 1, j, k + 1)

             # Try character from second string
            if j < len(s2) and s2[j] == s3[k]:
                result = result or dfs(i, j + 1, k + 1)
            
            memo[(i,j)] = result
            return result
        
        return dfs(0,0,0)
```

### Key Concepts:
- Top Down Memoization approach is easiest to understand for this problem
- Basecase: if we reach the end of all 3 strings, then we successfully interleaved the strings
- If k is out of bounds before i or j, then that means we could not find a solution so we return false. 
- We recursively try picking from the first or second string if we have a match. If neither string has a matching character then we return false. 

### Time and Space:
- Time: O(m * n)
- Space: O(m * n)

Note: Can be optimised with Bottom Up Space Optimised. 

## Longest Increasing Path in Matrix
You are given a 2-D grid of integers matrix, where each integer is greater than or equal to 0.

Return the length of the longest strictly increasing path within matrix.

From each cell within the path, you can move either horizontally or vertically. You may not move diagonally.

```python
# Find the longest increasing path in a matrix
# edgecase: check if valid matrix length
# initialise m, n, memo and directions
# dfs helper with row and col params
# if in memo return memo
# initialise max_length to 1 (this gives the current cell a value)
# loop through directions, create new_row, and new_col
# check inbounds, and if new cell is greater than old cell
# path length is 1 + dfs of new cell
# update max length to the max of itself and new path length
# save max length to memo, then return max length
# initialise result to 0, then iterate through matrix
# set result to the max of itself and the dfs at current cell
# return result
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        m, n = len(matrix), len(matrix[0])
        memo = {}
        directions = [(1,0), (0,1), (-1, 0), (0,-1)]

        def dfs(row, col):
            if (row, col) in memo:
                return memo[(row, col)]
            
            max_len = 1

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < m and 0 <= new_col < n and matrix[new_row][new_col] > matrix[row][col]:
                    max_len = max(max_len, 1 + dfs(new_row, new_col))
            
            memo[(row, col)] = max_len
            return max_len
        
        result = 0
        for i in range(m):
            for j in range(n):
                result = max(result, dfs(i, j))
        
        return result
```

### Key Concepts
- Can be solved using a Top Down Memoization approach
- Basecase: when we don't find another valid cell to add to our path
- We can solve this recursively by calling a DFS on each cell in the matrix
- We try all 4 directions if and check if they are in bounds or a strictly increasing value. 
- We use a memo to track the max path of each visited cell to avoid repeated work


### Time and Space:
- Time: O(m*n)
- Space: O(m*n)

## Distinct Subsequences
You are given two strings s and t, both consisting of english letters.

Return the number of distinct subsequences of s which are equal to t.

```python
# return the number of distinct subsequences 
# Top Down DP (Memoization) approach
# initialise memo
# helper dfs with i and j
# Basecase: if j equals len t, return 1
# if i equals len t then we didnt find a valid subsequence return 0
# check memo
# if characters are equal, 
# call dfs with i and j incremented, plus dfs of just i incremented
# else, call dfs of just i incremented
# store in memo and return 
# return dfs at 0,0
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        memo = {}

        def dfs(i, j):
            if j == len(t):
                return 1
            if i == len(s):
                return 0

            if (i, j) in memo:
                return memo[(i,j)]
            
            if s[i] == t[j]:
                result = dfs(i+1, j+1) + dfs(i+1, j)
            else:
                result = dfs(i+1, j)
            memo[(i,j)] = result
            return result
        
        return dfs(0,0)
```

### Key Concepts:
- Can be solved with Top Down DP (Memoization)
- Similar template to most Top down dp solutions
- Basecase: if j index equals length of target string, we found a subsequence return 1
- if i is length of input string, we want to return 0 we didnt find a subsequence
- Use a memo to prevent repeated work
- if characters are equal, we want to call dfs on both: moving i pointer, and moving both i and just
- if characters are not equal, we only want to call dfs on moving i pointer


### Time and Space:
Time: O(m*n)
Space: O(m*n)

## Edit Distance
You are given two strings word1 and word2, each consisting of lowercase English letters.

You are allowed to perform three operations on word1 an unlimited number of times:

Insert a character at any position
Delete a character at any position
Replace a character at any position
Return the minimum number of operations to make word1 equal word2.

```python
# return min operations needed to transform word1 to word 2
# We can use Top Down (Memoization)

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        memo = {}

        def dfs(i, j):
            if i == m: return n - j # "Deletes the remaining characters from this string"
            if j == n: return m - i # "Deletes the remaining characters from this string"

            if (i, j) in memo:
                return memo[(i, j)]
            
            if word1[i] == word2[j]:
                memo[(i,j)] = dfs(i + 1, j + 1)
            else:
                memo[(i, j)] = 1 + min(
                    dfs(i + 1, j), # Delete
                    dfs(i, j + 1), # Insert
                    dfs(i + 1, j + 1) # Replace
                )
            return memo[(i, j)]
            
        
        return dfs(0, 0)
```

### Key Concepts:
- Can be done using Top Down DP with Memoization
- Basecases: if we reach the end of one of the strings, we want to return the remaining length of the other string
- If both characters are equal then we "skip" them since no operations are needed
- else we want to take the minimum of 3 decisions: delete, replace, and insert + 1 cost

### Time and Space
- Time: O(m*n)
- Space: O(m*n)

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

## Burst Balloons
You are given an array of integers nums of size n. The ith element represents a balloon with an integer value of nums[i]. You must burst all of the balloons.

If you burst the ith balloon, you will receive nums[i - 1] * nums[i] * nums[i + 1] coins. If i - 1 or i + 1 goes out of bounds of the array, then assume the out of bounds value is 1.

Return the maximum number of coins you can receive by bursting all of the balloons.

```python
# Find the maximum points you can collect by bursting balloons
# Top Down DP Approach: Decide which balloon to burst last in each subarray. 
# Add boundary: [1] + nums + [1]
# initialise n and memo
# dp helper function with left and right pointers
# basecase: no balloons between left and right, return 0
# check memo
# init max points to 0
# try bursting each balloon k between left and right as the last one
# points = left subproblem + right subproblem + burst k last
# set max_points to the max of max_points and points, save in memo and return
# return result of dp where left = 0, and right = n - 1

class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        balloons = [1] + nums + [1]
        n = len(balloons)
        memo = {}

        def dp(left, right):
            if left + 1 == right:
                return 0
            
            if (left, right) in memo:
                return memo[(left, right)]

            max_points = 0

            # Try bursting every balloon k last
            for k in range(left + 1, right):
                points = (
                    dp(left, k) +
                    dp(k, right) + 
                    (balloons[left] * balloons[k] * balloons[right])
                )
                max_points = max(max_points, points)
            
            memo[(left, right)] = max_points
            return max_points
        
        return dp(0, n - 1)
```

### Key Concepts:
- The trick is to burst the current balloon last. 
- We can solve using Top Down DP (Memoization)
- Basecase: if there are no balloons between the left and right pointers we return -
- Maintain a max_points variable
- Try every k position between left and right pointers, if the max_points gets updated after trying every k, then we save it in the memo

### Time and Space:
- Time: O(n^3)
- Space: O(n^2)