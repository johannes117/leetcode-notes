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

- Time: O(m\*n)
- Space: O(m\*n), height of the callstack

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

- Time: O(m\*n)
- Space: O(m\*n), held in the recursive call stack.

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

- Time: O(m\*n)
- Space: O(m\*n)

## Rotting Oranges:

You are given a 2-D matrix grid. Each cell can have one of three possible values:

0 representing an empty cell
1 representing a fresh fruit
2 representing a rotten fruit
Every minute, if a fresh fruit is horizontally or vertically adjacent to a rotten fruit, then the fresh fruit also becomes rotten.

Return the minimum number of minutes that must elapse until there are zero fresh fruits remaining. If this state is impossible within the grid, return -1.

```python
# Return the minimum number of minutes that must elapse until there are zero fresh oranges remaining. If impossible -1
# visit every cell in grid, if rotten add to queue, if fresh increment fresh counter
# edgecase: if fresh 0 return 0
# initialise minute counter as -1
# while q, increment minute for each round/level
# each level popleft and call bfs on each direction
# basecases: out of bounds, cell not fresh. return
# replace cell with rotten, decrement fresh count, append to queue
# return minutes if freshcount is 0, else return -1
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        FRESH, ROTTEN, EMPTY = 1, 2, 0
        m, n = len(grid), len(grid[0])
        q = deque()
        fresh_count = 0

        for i in range(m):
           for j in range(n):
            if grid[i][j] == FRESH:
                fresh_count += 1
            elif grid[i][j] == ROTTEN:
                q.append((i, j))
        if fresh_count == 0:
            return 0

        minute_count = -1
        while q:
            minute_count += 1
            round_size = len(q)
            for _ in range(round_size):
                i, j = q.popleft()
                for r, c in [[i+1, j], [i-1, j], [i, j+1], [i, j-1]]:
                    if 0 <= r < m and 0 <= c < n and grid[r][c] == FRESH:
                        grid[r][c] = ROTTEN
                        q.append((r, c))
                        fresh_count -= 1

        if fresh_count == 0:
            return minute_count
        else:
            return -1
```

### Key Concepts:

- We can use BFS to calculate the rotten traversal throughout the grid.
- Similar solution to the Walls and Gates problem
- Maintain a minute count, each level/round we increment the minute count.
- We want to first count the number of fresh oranges, and add all of the rotten orange positions to the queue.
- For each level of the queue, we want to take the length of the queue at the beginning of the round and process only those number of items in the queue.
- This way that every item that is added during this round will be processed in the next round. Hence will be processed in the next minute.
- For each item in the queue, we want to check if its a valid cell, and whether its fresh, if so we add it to the queue, set it to rotten and decrememnt the fresh count.
- We want to return the minute count if the fresh count reached 0, else return -1.

# Time and Space Complexity

- Time: O(m\*n)
- Space: O(m\*n)

## Pacific Atlantic Water Flow

You are given a rectangular island heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

The islands borders the Pacific Ocean from the top and left sides, and borders the Atlantic Ocean from the bottom and right sides.

Water can flow in four directions (up, down, left, or right) from a cell to a neighboring cell with height equal or lower. Water can also flow into the ocean from cells adjacent to the ocean.

Find all cells where water can flow from that cell to both the Pacific and Atlantic oceans. Return it as a 2D list where each element is a list [r, c] representing the row and column of the cell. You may return the answer in any order.

```python
# m * n martrix heights. Find all cells where water can flow to both Pacific and Atlantic.
# return a 2d list
# define a queue and seen set for pacific and atlantic, define m and n
# Add the 1st row, 1st col, last row, last col to the queue and seen sets for atl and pac
# define a getCoords helper function with queue and seen as parameters
# initialise a new coords set
# while queue, we want to popleft from the queue.
# loop through all 4 directions (offset)
# conditions: inbounds, height of neighbor is bigger than current element, and neigbor is not in seen
# conditions met: add to seen and queue
# call getCoords for both atl and pac, return intersection of seen sets.
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        pac_queue, pac_seen = deque(), set()
        atl_queue, atl_seen = deque(), set()
        m, n = len(heights), len(heights[0])

        # Add First Row Pac
        for j in range(n):
            pac_queue.append((0, j))
            pac_seen.add((0, j))

        # Add First Column to Pac
        for i in range(1, m):
            pac_queue.append((i, 0))
            pac_seen.add((i, 0))

        # Add Last Row Atl
        for i in range(m):
            atl_queue.append((i, n - 1))
            atl_seen.add((i, n - 1))

        # Add Last Col to Atl
        for j in range(n - 1):
            atl_queue.append((m - 1, j))
            atl_seen.add((m - 1, j))

        def getCoords(que, seen):
            while que:
                i, j = que.popleft()
                for i_off, j_off in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                    r, c = i + i_off, j + j_off
                    if 0 <= r < m and 0 <= c < n and heights[r][c] >= heights[i][j] and (r, c) not in seen:
                        que.append((r, c))
                        seen.add((r, c))

        getCoords(pac_queue, pac_seen)
        getCoords(atl_queue, atl_seen)
        return list(pac_seen.intersection(atl_seen))
```

### Key Concepts:

- The first row, and first column will always be able to flow into the pacific
- The last row and last column will always be able to flow into the atlantic.
- Therefore we must perform a BFS search from these positions to find adjacent land that can flow into them (higher or same).
- We want to figure out all cells that can flow into Pacific, and all Cells that can flow into Atlantic and then take the intersection of the 2.

### Time and Space:

- Time: O(m\*n)
- Space: O(m\*n)

## Surrounded Regions
You are given a 2-D matrix board containing 'X' and 'O' characters.

If a continous, four-directionally connected group of 'O's is surrounded by 'X's, it is considered to be surrounded.

Change all surrounded regions of 'O's to 'X's and do so in-place by modifying the input board.

```python
# You are given a 2D grid mage of water (X) and islands of land (O). Find all islands that don't touch the border (ignoring diagonal) and sink them into water
# Scan the borders for land and use DFS to "shield" those islands )set O to T. Next sink the unshielded islands (sset ) to X). Finally remove the shields (Set T to O)
# 1. Capture unsurrounded regions (O -> T)
# 2. Capture surrounded regions (O -> X)
# 3. Uncapture unsurrounded regions (T -> O)
# DFS function with basecases: Out of bounds or position is not "O"
# If not basecase: change position to a "T", call DFS on all 4 directions. 
# To iterate through border positions use condition, if position is "O" and row is 0 or ROWS - 1 or column is 0 or cols - 1
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        ROWS, COLS = len(board), len(board[0])

        def dfs(r, c):
            # Base case: Out of bounds or not an O
            if r < 0 or c < 0 or r == ROWS or c == COLS or board[r][c] != "O":
                return
            board[r][c] = "T"
            dfs(r-1, c)
            dfs(r+1, c)
            dfs(r, c-1)
            dfs(r, c+1)
        
        # Capture unsurrounded regions (DFS)
        for r in range(ROWS):
            for c in range(COLS):
                if (board[r][c] == "O" and (r in [0, ROWS-1] or c in [0, COLS-1])):
                    dfs(r, c)

        # Capture surrounded regions
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "O":
                    board[r][c] = "X"

        # Uncapture unsurrounded regions
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "T":
                    board[r][c] = "O"
```

Slightly Optimised solution:
```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        ROWS, COLS = len(board), len(board[0])

        def capture(r, c):
            if (r < 0 or c < 0 or r == ROWS or 
                c == COLS or board[r][c] != "O"
            ):
                return
            board[r][c] = "T"
            capture(r + 1, c)
            capture(r - 1, c)
            capture(r, c + 1)
            capture(r, c - 1)

        for r in range(ROWS):
            if board[r][0] == "O":
                capture(r, 0)
            if board[r][COLS - 1] == "O":
                capture(r, COLS - 1)
        
        for c in range(COLS):
            if board[0][c] == "O":
                capture(0, c)
            if board[ROWS - 1][c] == "O":
                capture(ROWS - 1, c)

        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == "T":
                    board[r][c] = "O"
```

### Key Concepts:
- We want to "shield" islands that are touching the boarders, this means we need to iterate through all of the border positions and perform a DFS on all of the "O" positions. 
- When performing DFS on an "O" border position, we will effectively "shield" the entire connected island by turning them into "T" strings. 
- Later once we have shielded the islands touching the borders, we iterate through the grid and turn all remaining "O" cells into "X"s. Effectively "sinking" the islands. 
- Once we have sunken all surrounded islands, we want to unshield the "T" cells and turn them back into "O" cells. 

### Time and Space Complexity:
- Time: O(m*n)
- Space: O(m*n)


## Course Schedule
You are given an array prerequisites where prerequisites[i] = [a, b] indicates that you must take course b first if you want to take course a.

The pair [0, 1], indicates that must take course 1 before taking course 0.

There are a total of numCourses courses you are required to take, labeled from 0 to numCourses - 1.

Return true if it is possible to finish all courses, otherwise return false.

```python
# If we detect a cycle in the graph return False, else return True
# We can use an Adjacency List to handle the graph: a Dictionary where each node is a key, and each key can have multiple neighbors as values. g ={ 1: [0], 2: [1, 3]}
# We can build the adjacency list by looping through a,b in the input list and then appending the b value to 'a' key in the dictionary. 
# set constants for UNVISITED, VISITING, VISITED. create a states array of UNVISITED values the same length as the numCourses
# dfs helper: node as parameter, retrieve state from states list using node, 
# if visited, return true, if visiting, we have detected a cycle return false
# otherwise we want to set the current node in the states list as visiting. 
# loop through each neighbor in the node in the adj list and perform dfs on the neighbor nodes. 
# set the current node in the states list as visiting.
# return True (we didnt detect a cycle)
# loop throuh numCourses, and call dfs
# return true if we make it to the end. 
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g = defaultdict(list)
        for a, b in prerequisites:
            g[a].append(b)
        
        UNVISITED, VISITING, VISITED = 0, 1, 2
        states = [UNVISITED] * numCourses

        def dfs(node):
            state = states[node]
            if state == VISITED:
                return True
            elif state == VISITING:
                return False
            
            states[node] = VISITING

            for nei in g[node]:
                if not dfs(nei):
                    return False
            
            states[node] = VISITED
            return True

        for i in range(numCourses):
            if not dfs(i):
                return False
        
        return True
```

### Key Concepts:
- Multiple new concepts introduced in this question.
- Adjacency Lists: is a collection of lists representing a graph, where each node stores a list of its adjacent nodes. 
- Example implementation: `g = defaultdict(list)` where we can add edges with g[source].append(destination)
- Cycle Detection: Uses a three-state tracking system, UNVISITED (0), VISITING (1), VISITED(2)
- If we encounter VISITING node during traversal, we've found a cycle
- DFS for Cycle detection:
- Base case1: If node is VISITED: Return True (no cycle from this path)
= Base case 2: If node is VISITING: Return False (cycle detected)

### Time and Space:
- Time: O(N + E)
- Time: O(N + E)

## Course Schedule II
You are given an array prerequisites where prerequisites[i] = [a, b] indicates that you must take course b first if you want to take course a.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
There are a total of numCourses courses you are required to take, labeled from 0 to numCourses - 1.

Return a valid ordering of courses you can take to finish all courses. If there are many valid answers, return any of them. If it's not possible to finish all courses, return an empty array.

```python
# return a list of nodes representing the order to take the courses in if there are no cycles
# build up the graph
# define constants and variables
# define dfs helper function
# basecases: VISITING (false), VISITED (true)
# set current to visiting, and perform DFS on the neigbors (return false if cycle found)
# set to visited and append to order, then return true
# loop through numCourses and call dfs, if cycle return []
# return order
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        order = []
        g = defaultdict(list)
        for a, b in prerequisites:
            g[a].append(b)
        
        UNVISITED, VISITING, VISITED = 0,1,2
        states = [UNVISITED] * numCourses

        def dfs(i):
            state = states[i]
            if state == VISITING:
                return False
            elif state == VISITED:
                return True
            
            states[i] = VISITING

            for nei in g[i]:
                if not dfs(nei):
                    return False
            
            states[i] = VISITED
            order.append(i)
            return True
        
        for i in range(numCourses):
            if not dfs(i):
                return []
        
        return order
```

### Key Concepts:
- Same as Course Schedule 1, but we need append the course/node to an order list and return it at the end

### Time and Space Complexity:
- Time: O(N + E)
- Space: O(N + E)

## Graph Valid Tree
Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

```python
# Given n nodes, labeled from 0 to n - 1, and a list of undirected edges, check whether these edges make up a valid tree
# edge case, if not n return true
# build an adjacency list. unpack edgges and then append them to the adjacency list
# initialise a visit set
# dfs helper: 
# early return if node is in visited set
# add the node to visited set
# loop through neibors, if neighbor is prev, continue
# if not dfs return false
# return true if none of the neibors returned false
# outside: return dfs starting at 0 node, with prev value of -1 and if the visited set is the same len as n. 
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if not n:
            return False
        adj = {i:[] for i in range(n)}
        for n1, n2 in edges:
            adj[n1].append(n2)
            adj[n2].append(n1)

        visited = set()

        def dfs(node, prev):
            if node in visited:
                return False
            
            visited.add(node)
            for nei in adj[node]:
                if nei == prev:
                    continue
                if not dfs(nei, node):
                    return False
            return True
        
        return dfs(0, -1) and n == len(visited)
```

### Key Concepts:
- A Valid Tree cannot contain any cycles, and all nodes must be connected. 
- To check for cycles, we use a visited set when doing DFS. We use a Prev variable to track the last visted node to exclude it from the visited check
- To check that all nodes are connected, we can check if the visited set is the same length as n (number of nodes). This means all nodes were visited and have edges. 
- We use an adjacency list to structure this graph. 

### Time and Space Complexity
- Time: O(V + E)
- Space: O(V + E)
