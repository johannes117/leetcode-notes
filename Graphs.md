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

## Clone Graph

Given a node in a connected undirected graph, return a deep copy of the graph.

Each node in the graph contains an integer value and a list of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
The graph is shown in the test cases as an adjacency list. An adjacency list is a mapping of nodes to lists, used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

For simplicity, nodes values are numbered from 1 to n, where n is the total number of nodes in the graph. The index of each node within the adjacency list is the same as the node's value (1-indexed).

The input node will always be the first node in the graph and have 1 as the value.

#### Iterative Solution
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
# return a deep copy of the graph
# using iterative DFS for practice. 
# edge case check if node exists
# initialise starting node, old to new mapping (nodeMap), stack, visited set, and add the starting node to visited. 
# build nodeMap
# while stack: pop the node from the stack, add it to the nodeMap.
# loop through neighbors, if not in visited add to visited and append to stack
#
# loop through nodemap, loop through neigbors, create new neighbor and append it to the new_node.neighbors
# return the nodeMap[start]

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        
        start = node
        nodeMap = {}
        visited = set()
        visited.add(start)
        stack = [start]

        while stack:
            node = stack.pop()
            nodeMap[node] = Node(val=node.val)

            for nei in node.neighbors:
                if nei not in visited:
                    visited.add(nei)
                    stack.append(nei)

        for old_node, new_node in nodeMap.items():
            for nei in old_node.neighbors:
                new_nei = nodeMap[nei]
                new_node.neighbors.append(new_nei)
        
        return nodeMap[start]
```

#### Recursive Solution
```python
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        
        start = node
        nodeMap = {}
        visited = set()
        visited.add(start)

        def dfs(node):
            # Create new entry into NodeMap
            nodeMap[node] = Node(val=node.val)

            for nei in node.neighbors:
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei)

        dfs(start)

        for old_node, new_node in nodeMap.items():
            for nei in old_node.neighbors:
                new_nei = nodeMap[nei]
                new_node.neighbors.append(new_nei)
        
        return nodeMap[start]
```

### Key Concepts:
- We build a hashmap to store the old and new nodes using DFS
- Once we have the old to new map, we can iterate through the map and link the neighbors. 

### Time and Space Complexity:
- Time: O(V + E)
- Space: O(v)


## Islands and Treasure (Walls and Gates)
You are given a 
m
×
n
m×n 2D grid initialized with these three possible values:

-1 - A water cell that can not be traversed.
0 - A treasure chest.
INF - A land cell that can be traversed. We use the integer 2^31 - 1 = 2147483647 to represent INF.
Fill each land cell with the distance to its nearest treasure chest. If a land cell cannot reach a treasure chest than the value should remain INF.

Assume the grid can only be traversed up, down, left, or right.

Modify the grid in-place.

```python
# modify the grid in place with the distances to the nearest treasure (gate)
# we perform a BFS from each gate simultaneously, and set the distance from each gate in each land cell as we traverse. 
# Basecases: Out of bounds, visited and water cell (-1). We add the gates to visited so we dont have to check these. 
# We initialise the queue with all of the gates, loop through the grid and append each gate to the queue and mark as visited. 
# maintain a distance variable, loop while queue exists
# loop through current queue length
# popleft and then set the cell as the current distance variable (for the gates this will be 0, and they are processed in the first loop)
# we want to add each cell in all 4 directions to the queue, call the helper function with the base cases. add to visited and queue
# once we are finished with a single level, increase the distance. 
class Solution:
    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        ROWS, COLS = len(grid), len(grid[0])
        q = deque()
        visited = set()

        def addRow(r, c):
            if r < 0 or c < 0 or r >= ROWS or c >= COLS or (r,c) in visited or grid[r][c] == -1:
                return
            q.append([r,c])
            visited.add((r,c))
        
        # Add Gates/Treasure to queue
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 0:
                    q.append([r,c])
                    visited.add((r,c))

        # Process Queue
        distance = 0
        while q:
            for i in range(len(q)):
                r, c = q.popleft()
                grid[r][c] = distance
                addRow(r+1, c)
                addRow(r-1, c)
                addRow(r, c+1)
                addRow(r, c-1)
            distance += 1
```

### Key Concepts:
- To solve this problem we must do a simultaneous BFS on each gate starting position. 
- We can achieve this using a queue, and by incrementing the distance each time we process a queue "level"
- Base cases are: Out of Bounds, Visited or Water/Wall
- We check if the cell is valid, if it is its added to the queue to be processed in the next level. 
- Once each cells neighbors have been added to the queue, and their distance updated, we continue to the next level. 
- Eventually each BFS path will reach a basecase and return, and all cell distances will have been updated. 


### Time and Space Complexity:
- Time: O(m*n)
- Space: O(m*n)