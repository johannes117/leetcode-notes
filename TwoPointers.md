# Two Pointers

## 125. Valid Palindrome

### Problem Description
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

### Solution

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left_pointer = 0
        right_pointer = len(s) - 1

        while left_pointer < right_pointer:
            # Skip non-alphanumeric characters for left pointer
            while left_pointer < right_pointer and not s[left_pointer].isalnum():
                left_pointer += 1
            
            # Skip non-alphanumeric characters for right pointer
            while left_pointer < right_pointer and not s[right_pointer].isalnum():
                right_pointer -= 1
            
            # Compare characters
            if s[left_pointer].lower() != s[right_pointer].lower():
                return False
            
            left_pointer += 1
            right_pointer -= 1

        return True
```

### Key Concepts

Two Pointer Technique
- Uses two pointers: left_pointer and right_pointer
- Start at opposite ends of the string
- Move towards each other while comparing characters

Handling Non-Alphanumeric Characters
- `isalnum()` method checks if a character is alphanumeric
- Skip non-alphanumeric characters using nested while loops
- Ensures only valid characters are compared

Case Insensitive Comparison
- `.lower()` method converts characters to lowercase
- Allows comparison regardless of character case

Time and Space Complexity
- Time Complexity: O(n), where n is the length of the string
- Space Complexity: O(1), as we only use two pointers

Common Pitfalls to Avoid
- Forgetting to skip non-alphanumeric characters
- Not handling case sensitivity
- Incorrect pointer movement logic

Example Walkthrough
- Input: "A man, a plan, a canal: Panama"

- Start with left_pointer at 'A' and right_pointer at 'a'
- Skip non-alphanumeric characters
- Compare characters (case-insensitive)
- Move pointers towards each other
- Repeat until pointers meet or cross

Related Problems
- Two Sum
- Container With Most Water
- Trapping Rain Water

## 167. Two Sum II - Input Array Is Sorted

### Problem Description
Given a sorted array of integers, find two numbers that add up to a specific target number. Return the 1-indexed positions of these numbers.

The solution must use O(1) additional space and there is exactly one valid solution.

### Solution

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left_pointer = 0
        right_pointer = len(numbers) - 1

        while True:
            if numbers[left_pointer] + numbers[right_pointer] > target:
                right_pointer -= 1
            elif numbers[left_pointer] + numbers[right_pointer] < target:
                left_pointer += 1
            else:
                return [left_pointer + 1, right_pointer + 1]
```

### Key Concepts

Two Pointer Technique with Sorted Array
- Uses two pointers starting from opposite ends
- Leverages the sorted nature of the array
- Moves pointers based on sum comparison with target

Efficient Search Strategy
- If sum > target: decrease right pointer (reduces sum)
- If sum < target: increase left pointer (increases sum)
- If sum = target: found solution

Time and Space Complexity
- Time Complexity: O(n), where n is the length of the array
- Space Complexity: O(1), using only two pointers

Key Differences from Regular Two Sum
- Array is pre-sorted
- Returns 1-indexed positions
- Guaranteed exactly one solution
- Must use O(1) space (no hash map allowed)

Example Walkthrough
- Input: numbers = [1,2,3,4], target = 3
- Start with pointers at indices 0 and 3
- Compare sum with target and adjust pointers
- Return [1,2] when sum equals target

## 15. 3Sum

### Problem Description
Given an integer array nums, find all unique triplets in the array that sum to zero. The solution must not contain duplicate triplets.

### Solution
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        outputArr = []
        nums.sort()  # Sort array to handle duplicates effectively

        for i in range(len(nums)):
            # Skip duplicates for first number
            if i > 0 and nums[i] == nums[i-1]:
                continue
                
            left = i + 1
            right = len(nums) - 1
            
            while left < right:
                threeSum = nums[i] + nums[left] + nums[right]

                if threeSum == 0:
                    outputArr.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    # Skip duplicates for second number
                    while nums[left] == nums[left - 1] and left < right: 
                        left += 1
                    # Skip duplicates for third number
                    while nums[right] == nums[right + 1] and left < right: 
                        right -= 1
                elif threeSum > 0:
                    right -= 1
                elif threeSum < 0:
                    left += 1
```

### Key Concepts

Extended Two Pointer Technique
- Uses a fixed pointer (i) and two moving pointers (left, right)
- Sorts array first to enable duplicate handling and efficient search
- Moves pointers based on sum comparison with zero

Duplicate Handling
- Skip duplicates for first number using the outer loop
- Skip duplicates for second and third numbers after finding a valid triplet
- Ensures unique triplets in output

Efficient Search Strategy
- If sum > 0: decrease right pointer (reduces sum)
- If sum < 0: increase left pointer (increases sum)
- If sum = 0: found valid triplet, handle duplicates and continue search

Time and Space Complexity
- Time Complexity: O(n²), where n is the length of the array
- Space Complexity: O(1) excluding the output array

Common Pitfalls to Avoid
- Not handling duplicates properly
- Forgetting boundary checks (left < right)
- Incorrect pointer movement after finding a valid triplet
- Not sorting the array first

Example Walkthrough
Input: nums = [-1,0,1,2,-1,-4]
1. Sort array: [-4,-1,-1,0,1,2]
2. Fix first number, use two pointers for remaining elements
3. Skip duplicates to avoid repeated triplets
4. Output: [[-1,-1,2],[-1,0,1]]

Related Problems
- Two Sum
- 4Sum
- 3Sum Closest

## 11. Container With Most Water

### Problem Description
Given an array of heights representing bars, find two bars that together with the x-axis forms a container that can hold the maximum amount of water.

### Solution
```python
class Solution:
    def maxArea(self, heights: List[int]) -> int:
        maxArea = 0
        l = 0
        r = len(heights) - 1

        while l < r:
            # Calculate area using width × height
            conLen = r - l
            conHeight = min(heights[l], heights[r])
            area = conLen * conHeight
            
            maxArea = max(area, maxArea)
            
            # Move pointer of smaller height inward
            if heights[l] < heights[r]:
                l += 1
            else:
                r -= 1

        return maxArea
```

### Key Concepts

Two Pointer Technique
- Uses two pointers starting from opposite ends
- Area is calculated using width (distance between pointers) × height (minimum of two bar heights)
- Move the pointer with smaller height inward to potentially find larger area

Efficient Search Strategy
- No need to check all combinations (would be O(n²))
- Moving the pointer with larger height would only decrease the area since:
  1. Width would decrease
  2. Height would be limited by the smaller bar anyway

Time and Space Complexity
- Time Complexity: O(n), where n is the length of the array
- Space Complexity: O(1), using only two pointers

Common Pitfalls to Avoid
- Not moving pointers correctly after area calculation
- Using incorrect area calculation formula
- Not considering that container height is limited by shorter bar

Example Walkthrough
Input: height = [1,7,2,5,4,7,3,6]
- Start with pointers at both ends
- Calculate area as (right - left) × min(height[left], height[right])
- Move pointer with smaller height inward
- Keep track of maximum area seen so far

Related Problems
- Trapping Rain Water
- Two Sum
- 3Sum


## 42. Trapping Rain Water

### Problem Description
Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

### Solution
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height: return 0

        maxL = height[0]
        maxR = height[len(height) - 1]

        l, r = 0, len(height) - 1

        totalArea = 0
        while l < r:
            if maxL < maxR:
                l += 1
                maxL = max(maxL, height[l])
                totalArea += maxL - height[l]
            else:
                r -= 1
                maxR = max(maxR, height[r])
                totalArea += maxR - height[r]
        
        return totalArea
```

### Key Concepts

Two Pointer Technique
- Uses two pointers starting from opposite ends
- Maintains maximum height seen from left and right sides
- Water trapped at each position is determined by the smaller of maxL and maxR

Efficient Water Calculation
- For each position, water trapped = min(maxL, maxR) - height[current]
- Move pointer on the side with smaller max height
- Update max height after moving pointer
- Add trapped water for current position

Time and Space Complexity
- Time Complexity: O(n), where n is the length of the array
- Space Complexity: O(1), using only constant extra space

Common Pitfalls to Avoid
- Not handling empty input array
- Incorrect pointer movement logic
- Not updating max heights at the right time
- Not considering that water level is determined by smaller of the two max heights

Example Walkthrough
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
- Start with pointers at ends
- Track maximum heights from both sides
- Move pointer from side with smaller max height
- Add trapped water at each step
- Output: 6 units of water

Related Problems
- Container With Most Water
- Two Sum
- Product of Array Except Self
