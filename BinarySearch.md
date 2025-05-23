# Binary Search

## Binary Search
You are given an array of distinct integers nums, sorted in ascending order, and an integer target.

Implement a function to search for target within nums. If it exists, then return its index, otherwise, return -1.

Your solution must run in O(logn) time.

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            m = l + ((r - l) // 2)
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        return -1
```

### Key Concepts
- Binary search is a divide and conquer algorithm that works by repeatedly dividing the search interval in half.
- The search interval is halved by calculating the middle index.
- If the target is found, the function returns the index of the target.
- If the target is not found, the function returns -1.


# 2D Matrix Binary Search
You are given an m x n 2-D integer array matrix and an integer target.

Each row in matrix is sorted in non-decreasing order.
The first integer of every row is greater than the last integer of the previous row.
Return true if target exists within matrix or false otherwise.

Can you write a solution that runs in O(log(m * n)) time?

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix) # returns the length of 
        n = len(matrix[0]) # Returns the length of columns
        l = 0
        r = m * n - 1

        while l <= r:
            mid = (l + r) // 2
            i = mid // n # Divide middle by the length of columns
            j = mid % n # Mid Modulo length of columns gives us the j index

            mid_val = matrix[i][j]

            # Determine if target is value
            if mid_val == target:
                return True
            elif mid_val < target:
                # move left pointer
                l = mid + 1
            else:
                r = mid - 1
        
        return False
```

### Key Concepts
- We can treat the 2D matrix as a 1D array and apply binary search on it.
- We can calculate the middle index of the 1D array and then find the corresponding row and column in the 2D matrix.
- We can then apply binary search on the row to find the target.
- To calculate the middle index of the 1D array, we can use the formula `mid = (L + R) // 2` where `L` is the leftmost index and `R` is the rightmost index.
- To find the corresponding row and column in the 2D matrix, we can use the formula `row = mid // n` and `col = mid % n` where `n` is the number of columns in the 2D matrix.
- Then we can get the value at the corresponding row and column in the 2D matrix using the formula `matrix[row][col]`.
- We can then compare the value with the target and adjust the search interval accordingly.
- We can repeat this process until we find the target or the search interval is empty.
- If the target is not found, we return -1.
- To get the rightmost index of the 1D array, we can use the formula `R = len(matrix) * len(matrix[0]) - 1`.

### Time Complexity
- O(log(m * n)) time
- We are performing binary search on a 1D array which takes O(log(m * n)) time.

### Space Complexity
- O(1) space
- We are not using any extra space to store the result. Two pointers are used to store the search interval.

## Koko Eating Bananas
You are given an integer array piles where piles[i] is the number of bananas in the ith pile. You are also given an integer h, which represents the number of hours you have to eat all the bananas.

You may decide your bananas-per-hour eating rate of k. Each hour, you may choose a pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, you may finish eating the pile but you can not eat from another pile in the same hour.

Return the minimum integer k such that you can eat all the bananas within h hours.

### Key Concepts
- We can utilise binary search to improve the time complexity of the solution.
- Use two pointers, `l` and `r`, to represent the search interval.
- r pointer can be calculated by taking the max value in the piles array. max(piles)
- we can set l pointer to 1 because koko can eat at least 1 banana per hour.
- we can set r pointer to the max value in the piles array because koko can eat at most the max value in the piles array bananas per hour.
- we can calculate the middle value of the search interval using the formula `mid = (l + r) // 2`.
- we can calculate the total time taken to eat all the bananas using the formula `total_time = sum(math.ceil(piles[i] / mid) for i in range(len(piles)))`.
- we can compare the total time taken to eat all the bananas with the given hours `h` to determine if the current middle value is too fast or too slow.
- if the total time taken is greater than `h`, we can move the left pointer to `mid + 1` to search for a faster eating rate.
- if the total time taken is less than or equal to `h`, we can move the right pointer to `mid - 1` to search for a slower eating rate.
- we can repeat this process until the left pointer is greater than the right pointer.
- the minimum integer k such that you can eat all the bananas within h hours is the left pointer.

```python
    class Solution:
        def minEatingSpeed(self, piles: List[int], h: int) -> int:
            # Define left and right pointers
            l, r = 1, max(piles) # right pointer is the maximum value in the input array
            res = 0

            while l <= r:
                # calculate middle value
                mid = (l + r) // 2
                totalTime = 0
                # use mid value to calculate amount of hours it will take to eat all of the bananas
                for pile in piles:
                    # Converts pile to float, divides by eating speed, rounds up to nearest hour since partial hours invalid
                    totalTime += math.ceil(float(pile) / mid)              
                if totalTime <= h:
                    # That means we found a valid candidate, and we need to check values to the left of the mid pointer
                    res = mid
                    r = mid - 1
                else:
                    # we did not find a valid candidate, we need to increase the amount of bananas per hour (check right of mid)
                    l = mid + 1
            
            return res
```

### Time Complexity
- O(n log m) time
- We are performing binary search on the search interval which takes O(log m) time.
- We are also iterating through the piles array which takes O(n) time.
- Therefore, the time complexity is O(n log m) time.

### Space Complexity
- O(1) space
- We are not using any extra space to store the result. Two pointers are used to store the search interval.

## Find Minimum in Rotated Sorted Array
You are given an array of length n which was originally sorted in ascending order. It has now been rotated between 1 and n times. For example, the array nums = [1,2,3,4,5,6] might become:

[3,4,5,6,1,2] if it was rotated 4 times.
[1,2,3,4,5,6] if it was rotated 6 times.
Notice that rotating the array 4 times moves the last four elements of the array to the beginning. Rotating the array 6 times produces the original array.

Assuming all elements in the rotated sorted array nums are unique, return the minimum element of this array.

A solution that runs in O(n) time is trivial, can you write an algorithm that runs in O(log n) time?

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # if value at middle pointer is greater or equal to value at left pointer search Right
        # else search left. 
        l, r = 0, len(nums) - 1
        res = nums[0]

        while l <= r:
            if nums[l] < nums[r]:
                res = min(nums[l], res)
                break
            mid = (l + r) // 2
            res = min(res, nums[mid])
            if nums[mid] >= nums[l]:
                # we want to search right
                l = mid + 1
            else:
                # we want to search left
                r = mid - 1
        
        return res
```

### Key Concepts
- We can utilise binary search to improve the time complexity of the solution.
- We can use two pointers, `l` and `r`, to represent the search interval.
- We can calculate the middle index of the search interval using the formula `mid = (l + r) // 2`.
- We can compare the value at the middle index with the value at the left pointer to determine if the minimum value is to the left or right of the middle index.
- If the value at the middle index is greater than or equal to the value at the left pointer, we can move the left pointer to `mid + 1` to search for the minimum value to the right of the middle index.
- If the value at the middle index is less than the value at the left pointer, we can move the right pointer to `mid - 1` to search for the minimum value to the left of the middle index.
- We can repeat this process until the left pointer is greater than the right pointer.
- The minimum value in the rotated sorted array is the value at the left pointer.
- There is a special case where the array is not rotated, in this case, we can return the first element of the array.


# Search in Rotated Sorted Array
You are given an array of length n which was originally sorted in ascending order. It has now been rotated between 1 and n times. For example, the array nums = [1,2,3,4,5,6] might become:

[3,4,5,6,1,2] if it was rotated 4 times.
[1,2,3,4,5,6] if it was rotated 6 times.
Given the rotated sorted array nums and an integer target, return the index of target within nums, or -1 if it is not present.

You may assume all elements in the sorted rotated array nums are unique,

A solution that runs in O(n) time is trivial, can you write an algorithm that runs in O(log n) time?

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # Use binary search to find pivot
        # once you've found the pivot, do a binary search on either the left sorted portion or right sorted portion. 
        # Handle condition where l < r (array is not rotated)

        # Finding the pivot
        l, r = 0, len(nums) - 1

        while l < r:
            # find m
            m = (l + r) // 2
            # if m greater than nums[r] increment left pointer
            if nums[m] > nums[r]: # the middle pointer being larger than right pointer suggest the pivot is to the right. 
                l = m + 1
            # else increment right pointer
            else:
                r = m

        # left pointer will be at the pivot, save pivot
        pivot = l

        # 3 conditions to set the new pointers
        # pivot index is 0, simply perform binary search on the entire array
        if nums[pivot] == nums[0]:
            l, r = 0, len(nums) - 1
        # target is to the left of the pivot
        elif target >= nums[0]:
            l, r = 0, pivot - 1
        # target is to the right of the pivot
        else:
            l, r = pivot, len(nums) - 1

        # perform a traditional binary search using the new pointers
        while l <= r:
            m = (l + r) // 2
            if nums[m] == target:
                return m
            if target < nums[m]:
                r = m - 1
            else:
                l = m + 1

        # return value
        return -1
```

### Key Concepts
- We need to perform 2 binary searches. One to find the pivot and one to find the target.
- You can find the pivot by converging the left and right pointers until they are equal.
- Once you've found the pivot, you can perform a binary search on either the left sorted portion or right sorted portion.
- There are 3 conditions to set the new pointers:
    - pivot index is 0, simply perform binary search on the entire array
    - target is to the left of the pivot
    - target is to the right of the pivot
- Perform a traditional binary search using the new pointers.

# Time Based Key-Value Store
Implement a time-based key-value data structure that supports:

Storing multiple values for the same key at specified time stamps
Retrieving the key's value at a specified timestamp
Implement the TimeMap class:

TimeMap() Initializes the object.
void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
String get(String key, int timestamp) Returns the most recent value of key if set was previously called on it and the most recent timestamp for that key prev_timestamp is less than or equal to the given timestamp (prev_timestamp <= timestamp). If there are no values, it returns "".
Note: For all calls to set, the timestamps are in strictly increasing order.

```python
class TimeMap:

    def __init__(self):
        # initialise hashmap - key: list of [val, timestamp]
        self.keyStore = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        # check if key exists, if not, initialise to empty list
        if key not in self.keyStore:
            self.keyStore[key] = []
        # append [value, timestamp] to key value
        self.keyStore[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:
        # initialise response as empty string, initialise values using key with a default value of []
        res = ""
        values = self.keyStore.get(key,[])

        #perform binary search
        l, r = 0, len(values) - 1
        while l <= r:
            m = (l + r) // 2
            # Key cases: if values[m][1] == timestamp, return values[m][0]
            if values[m][1] == timestamp:
                return values[m][0]
            # if values[m][1] less than timestamp, increment left pointer. Set response to values[m][0]
            elif values[m][1] < timestamp:
                res = values[m][0]
                l = m + 1
            # else, decrement right pointer. 
            else:
                r = m - 1
        # return response
        return res
```

### Key Concepts
- We need to perform a binary search to find the most recent value of key if set was previously called on it and the most recent timestamp for that key prev_timestamp is less than or equal to the given timestamp (prev_timestamp <= timestamp).
- We can use a hashmap to store the key and its values and timestamps.
- We can use a binary search to find the most recent value of key if set was previously called on it and the most recent timestamp for that key prev_timestamp is less than or equal to the given timestamp (prev_timestamp <= timestamp).
- We could use a defaultdict to simplify the code, instead of checking if the key exists, we can just append the value and timestamp to the list.