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

## Happy Number
A non-cyclical number is an integer defined by the following algorithm:

Given a positive integer, replace it with the sum of the squares of its digits.
Repeat the above step until the number equals 1, or it loops infinitely in a cycle which does not include 1.
If it stops at 1, then the number is a non-cyclical number.
Given a positive integer n, return true if it is a non-cyclical number, otherwise return false.

```python
# Return true if the number n is "happy", which means it will eventually reach 1, If we detect a cycle it is not happy
# Helper function to get the sum of squares:
#   - total variable, loop while input num is greater than 0
#   - digit = num % 10 (number modulo 10) (Gives us the right most digit)
#   - increment total by the digit * digit (squared)
#   - num //= 10 (divide num by 10, drop the rightmost digit)
#   - return total
# Initialise a seen set, 
# loop while n != 1, and not in seen
# add n to seen, and call helper on n
# return whether n == 1
class Solution:
    def isHappy(self, n: int) -> bool:
        
        def helper(num):
            total = 0
            while num > 0:
                digit = num % 10 # Get rightmost digit
                total += digit * digit # add digit squared to sum
                num //= 10 # drop rightmost digit
            return total

        seen = set()

        while n != 1 and n not in seen:
            seen.add(n)
            n = helper(n)
        
        return n == 1
```

### Key Concepts:
- essentially this is a cycle detection problem. Thats why we can solve it using a hashmap seen set, and store each computed sum of squares value. 
- Each time we calculate a sum of squares, we add that number to the seen set and update n.
- If n ever reaches 1, we know its a happy number
- If we break the loop, and n does not equal 1, we have detected a loop, because it means we have calculated a value that has been seen before, thus detecting a cycle. 
- Calculating the sum of squares: Get the rightmost digit using modulo 10, add the digits square to the total sum, and divide the current num by 10 (drop the rightmost digit)

### Time and Space:
- Time: O(log n), number of digits in worst case cycle
- Space: O(log n), storing seen numbers

## Pow(x, n)
Pow(x, n) is a mathematical function to calculate the value of x raised to the power of n (i.e., x^n).

Given a floating-point value x and an integer value n, implement the myPow(x, n) function, which calculates x raised to the power n.

You may not use any built-in library functions.

```python
# Compute x to the power n
# initialise memo
# helper function: base and exponent as parameters
# basecases: if exponent is 0, return 1, if 1 return base
# check memo
# recursive case: 
# if exponent is even, calculate halfpower using helper, base, expo // 2
# result is halfpower * halfpower
# else, odd component, result is base * helper, base, exp - 1
# store result in memo
# return result
# handle negatives: if n < -, return 1 over helper x, -n
# else return helper x, n
class Solution:
    def myPow(self, x: float, n: int) -> float:
        memo = {}

        def helper(base, exp):
            # Basecases:
            if exp == 0:
                return 1
            if exp == 1:
                return base
            if exp in memo:
                return memo[exp]
            
            if exp % 2 == 0:
                half_power = helper(base, exp // 2)
                result = half_power * half_power
            else:
                result = base * helper(base, exp - 1)
            
            memo[exp] = result
            return memo[exp]
        
        if n < 0:
            return 1 / helper(x, -n)
        else:
            return helper(x, n)
```

### Key Concepts:
- We can solve this using recursion with memoization to improve the time complexity. 
- basecases, if exponent is 0 return 1, if exponent is 1 return base. 
- if exponent in memo return memo
- Recursive: if exponent is even, just call helper on exponent // 2, then multiply the half powers together to get the result
- Else if odd, we want to call the helper on exponent - 1, and then multiply the result by the base (which is essentially exponent == 1)
- store in memo
- return memo
- Key insight for negatives, negative exponent is just positive exponent, but as the denominator. 
- 2^-2 is 1/2^2

### Time and Space
- Time: O(log n)
- Space: O(log n), call stack