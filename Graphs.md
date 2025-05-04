# Graphs

## Number of Islands

Given a 2D grid grid where '1' represents land and '0' represents water, count and return the number of islands.

An island is formed by connecting adjacent lands horizontally or vertically and is surrounded by water. You may assume water is surrounding the grid (i.e., all the edges are water).

```python
# given a 2d binary grid, count and return the number of islands
# define m and n, and number of islands
# check each starting position in grid: if position is "1" we want to increment islands counter and call dfs on that position
# dfs with i and j as parameters:
# Basecases: out of bounds and position is not "1"
# set the current position to "0", call dfs on all 4 directions
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        num_islands = 0

        def dfs(i, j):
            # Base case
            if i < 0 or j < 0 or i >= m or j >=n or grid[i][j] != '1':
                return
            else:
                grid[i][j] = '0'
                dfs(i+1, j)
                dfs(i-1, j)
                dfs(i, j+1)
                dfs(i, j-1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    num_islands += 1
                    dfs(i, j)

        return num_islands
```

### Key Concepts:

- Visit every starting position to find the top left of an island
- If we find an island we cincrement the island counter and call dfs on that position
- The dfs function essentially destroys that island, so that if we visit that position later, we don't count it again.
- Base cases: Out of bounds and if the current position is not '1'/land
- We want to destroy that land/mark it as 0 and then call dfs in all 4 directions.

### Time and Space Complexity:

- Time: O(m*n)
- Space: O(m*n), height of the callstack

Max Area of Island
You are given a matrix grid where grid[i] is either a 0 (representing water) or 1 (representing land).

An island is defined as a group of 1's connected horizontally or vertically. You may assume all four edges of the grid are surrounded by water.

The area of an island is defined as the number of cells within the island.

Return the maximum area of an island in grid. If no island exists, return 0.

```python
# m * n binary matrix grid where grid[i] is either 0 or 1 representing land. return the maximum area of an island. 
# visit every starting position and call dfs on positions that are 1
# dfs function with i and j as input (row and column)
# base case: out of bounds and position is not equal to 1. return 0
# if land we want to set to 0 and return the dfs call of all 4 positions + 1

class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        def dfs(r, c):
            if r < 0 or c < 0 or r >= m or c >= n or grid[r][c] != 1:
                return 0
            else:
                grid[r][c] = 0
                return 1 + dfs(r+1, c) + dfs(r-1, c) + dfs(r, c+1) + dfs(r, c-1)

        max_area = 0
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1:
                    max_area = max(max_area, dfs(r, c))
        return max_area
```

### Key Concepts:
- Similar to the Number of islands solution except we maintain a max area variable. 
- Basecases: Out of bounds and position is not land
- If we find a position that is land, we turn it into water and then return 1 + the recursive call in all 4 directions from that position. 

### Time and Space:
- Time: O(m*n)
- Space: O(m*n), held in the recursive call stack. 