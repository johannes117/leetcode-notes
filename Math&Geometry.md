# Math and Geometry

## Rotate Image
Given a square n x n matrix of integers matrix, rotate it by 90 degrees clockwise.

You must rotate the matrix in-place. Do not allocate another 2D matrix and do the rotation.

```python
# rotate the matrix 90 degrees in place
# define n
# Step 1: Transpose the matrix, nested for loop, inner loop start from i to avoid swapping back
# Step 2: Reverse each row, loop through rows, call matrix[i].reverse()
# 
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)

        # Transpose:
        for i in range(n):
            for j in range(i, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Reverse
        for i in range(n):
            matrix[i].reverse()
```

### Key Concepts:
- To Rotate a Matrix 90 degrees, we have to perform, a transpose then a reversal. 
- Transpose is simply iterating through the matrix, and swapping the ith and jth values. 
- Reversing can be done using the .reverse() function in python. 

### Time and Space
- Time: O(n^2)
- Space: O(1)

## Spiral Matrix
Given an m x n matrix of integers matrix, return a list of all elements within the matrix in spiral order.

```python
# return a list of all elements in the matrix in spiral order:
# Intuition: Define 4 boundaries, and shift them inwards when we are done with that row or col
# Edgecase: if matrix has 0 rows or cols
# define, result list, top, bottom, left and right boundaries. 
# loop while top less than or equal to bottom, and left less than or equal to right
# 1. Traverse right along the top row. for column from left boundary to right boundary + 1. 
# append value at top, col
# shrink top
# 2. Traverse down along the right column: from top to bottom + 1, append value at row, right. Shrink boundary
# 3. Traverse left along the bottom row (if we still have rows): right to left - 1 backwards. append bottom, col, shrink boundary
# 4. Traverse up if we still have columns, from bottom to top - 1, backwards, append row, left, shrink boundary
# return result
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return []
        
        result = []
        top = 0
        right = len(matrix[0]) - 1
        left = 0
        bottom = len(matrix) - 1

        while top <= bottom and left <= right:
            # Traverse right
            for col in range(left, right + 1):
                result.append(matrix[top][col])
            top += 1

            # Traverse down
            for row in range(top, bottom + 1):
                result.append(matrix[row][right])
            right -= 1

            # Traverse left, if possible
            if top <= bottom:
                for col in range(right, left - 1, - 1):
                    result.append(matrix[bottom][col])
                bottom -= 1

            # Traverse up, if possible
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    result.append(matrix[row][left])
                left += 1
        
        return result
```

### Key Concepts:
- Maintain Top, Left, Right and Bottom bounaries.
- Traverse through the top boundary first, then shrink that boundary. So on until either Top meets bottom or left meets right boundaries

### Time and Space:
- Time: O(m*n)
- Space: O(m*n)

## Set Matrix Zeroes
Given an m x n matrix of integers matrix, if an element is 0, set its entire row and column to 0's.

You must update the matrix in-place.

Follow up: Could you solve it using O(1) space?

```python
# if an element is 0, set its entire row and column to 0's
# Traverse through the matrix twice, once to set the target positions with a *, and once to update the *'s with 0s
# loop through matrix, if position is 0, call helper function in all 4 directions
# loop through matrix, if position is *, set to 0
# helper: if in bounds, and position != 0, set to *
# call helper in the same direction. 
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        
        def helper(m, n, direction):
            if 0 <= m < len(matrix) and 0 <= n < len(matrix[0]) and matrix[m][n] != 0:
                matrix[m][n] = "*"
                if direction == "U":
                    helper(m-1, n, "U")
                elif direction == "D":
                    helper(m+1, n, "D")
                elif direction == "L":
                    helper(m, n-1, "L")
                else:
                    helper(m, n+1, "R")
        
        for m in range(len(matrix)):
            for n in range(len(matrix[0])):
                if matrix[m][n] == 0:
                    helper(m-1, n, "U")
                    helper(m+1, n, "D")
                    helper(m, n-1, "L")
                    helper(m, n+1, "R")
        
        for m in range(len(matrix)):
            for n in range(len(matrix[0])):
                if matrix[m][n] == "*":
                    matrix[m][n] = 0
```

### Time and Space:
- Time: O(m*n)
- Space: O(max(m, n))