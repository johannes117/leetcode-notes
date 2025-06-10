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