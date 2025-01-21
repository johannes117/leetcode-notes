# Two Pointers

## Best Time to Buy and Sell Stock


### Problem Description
You are given an integer array prices where prices[i] is the price of NeetCoin on the ith day.

You may choose a single day to buy one NeetCoin and choose a different day in the future to sell it.

Return the maximum profit you can achieve. You may choose to not make any transactions, in which case the profit would be 0.

### Solution
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l, r = 0, 1
        maxProfit = 0

        while r < len(prices):
            if prices[r] > prices[l]:
                currentProfit = prices[r] - prices[l]
                maxProfit = max(maxProfit, currentProfit)
            else:
                l = r
            r += 1
        
        return maxProfit
```

### Key Concepts
- Two pointers are used to iterate through the array
- The left pointer is used to buy the stock
- The right pointer is used to sell the stock
- The max profit is updated whenever the right pointer is greater than the left pointer
- The left pointer is updated whenever the right pointer is less than the left pointer
- The right pointer is incremented to move to the next day

This is a sliding window problem because we are using two pointers to iterate through the array.

## Longest Substring Without Repeating Characters

Given a string s, find the length of the longest substring without duplicate characters.

A substring is a contiguous sequence of characters within a string.

### Solution
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0

        l, r = 0, 1
        maxLen = 1
        # Set or Hashmap to store substring values.
        hashSet = set()
        hashSet.add(s[l])

        while r < len(s):
            if s[r] not in hashSet:
                hashSet.add(s[r])
                r += 1
            else:
                hashSet.remove(s[l])
                l += 1    
            
            maxLen = max(maxLen, len(hashSet))

        return maxLen
```

### Key Concepts
- Two pointers are used to iterate through the string
- The left pointer is used to remove the character from the set
- The right pointer is used to add the character to the set
- The max length is updated whenever the right pointer is greater than the left pointer
- The left pointer is incremented to move to the next character
- The right pointer is incremented to move to the next character

This is a sliding window problem because we are using two pointers to iterate through the string.
