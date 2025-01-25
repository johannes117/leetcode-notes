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

## Permutation in String

You are given two strings s1 and s2.

Return true if s2 contains a permutation of s1, or false otherwise. That means if a permutation of s1 exists as a substring of s2, then return true.

Both strings only contain lowercase letters.

### Solution
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2): return False

        # Initialise Frequency Arrays
        s1_freq = [0] * 26
        s2_freq = [0] * 26

        # Populate Frequency arrays
        for i in range(len(s1)):
            s1_freq[ord(s1[i]) - ord('a')] += 1 # Fully calculates freq array of s1
            s2_freq[ord(s2[i]) - ord('a')] += 1 # Calcs freq array of s2 in the first window of len(s1)

        if s1_freq == s2_freq:
            return True

        for i in range(len(s1), len(s2)):
            # Slide the Window
            s2_freq[ord(s2[i]) - ord('a')] += 1 # Adds a character
            s2_freq[ord(s2[i-(len(s1))]) - ord('a')] -= 1 # Removes far left character

            if s1_freq == s2_freq:
                return True
        
        return False
```

### Key Concepts
- Frequency arrays are used to store the frequency of each character in the strings
- The frequency arrays are compared to check if a permutation of s1 exists as a substring of s2
- The frequency arrays are updated by sliding the window
- The window is slid by adding the current character and removing the far left character
- The frequency arrays are compared to check if a permutation of s1 exists as a substring of s2
