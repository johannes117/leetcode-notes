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