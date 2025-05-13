# Advanced Graphs

## Network Delay Time
You are given a network of n directed nodes, labeled from 1 to n. You are also given times, a list of directed edges where times[i] = (ui, vi, ti).

ui is the source node (an integer from 1 to n)
vi is the target node (an integer from 1 to n)
ti is the time it takes for a signal to travel from the source to the target node (an integer greater than or equal to 0).
You are also given an integer k, representing the node that we will send a signal from.

Return the minimum time it takes for all of the n nodes to receive the signal. If it is impossible for all the nodes to receive the signal, return -1 instead.

```python
# n nodes, a list of travel times as directed edges: times[i] = (ui, vi, wi)
# build an adjacency list using u as the key and v and time as a tuple value
# initialise min_times and min_heap. min_heap is a list of tuples where (distance from source to node, node)
# while heap
# unpack distance and node
# if node in min_times ignore it
# store the distance in min_times using the node as the key (this is the shortest distance to reach this node)
# loop through neigbours and neighbor times in the current node
# if neighbor not in min_times, push to the heap: total distance + neighbor time, neighbor). 
# return the max of the min_times values, if min_times is length n, else return -1
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        graph = defaultdict(list)
        for u, v, t in times:
            graph[u].append([v, t])
        
        min_times = {}
        min_heap = [(0, k)] # (distance to node from k, node)

        while min_heap:
            distance_to_k, node = heapq.heappop(min_heap)
            if node in min_times:
                continue
            
            min_times[node] = distance_to_k

            for nei, nei_time in graph[node]:
                if nei not in min_times:
                    heapq.heappush(min_heap, (distance_to_k + nei_time, nei))
        
        if len(min_times) == n:
            return max(min_times.values())
        else:
            return -1
```

### Key Concepts:
- Dijkstras Algorithm: Dijkstra's algorithm iteratively explores the graph, maintaining the shortest distance found so far to each node and updating it whenever a shorter path is discovered through an unvisited neighbor.
- In this specific algorithm, we use a min heap to get the current shortest distance from k for each iteration. 
- If the node is not yet in the min_times dictionary, we know we have the shortest path to that node. 
- We want to add all of that nodes neighbors to the heap to be processed later. 
- If a neighbor is added to the heap, but its already in the min_times dictionary, we know we have already found a shorter distance to that node. 

### Time and Space Complexity 
- Time: O(E log V), where E is the number of edges and V is the number of vertices. The while loop runs ar most E times, and each heappush and heappop operation on a heap of up to v takes O(log V) time. 
- Space: O(V + E), where V is for the min_times dict, and the heap, and E is for the graph adjacency list. 


## Reconstruct Flight Path
You are given a list of flight tickets tickets where tickets[i] = [from_i, to_i] represent the source airport and the destination airport.

Each from_i and to_i consists of three uppercase English letters.

Reconstruct the itinerary in order and return it.

All of the tickets belong to someone who originally departed from "JFK". Your objective is to reconstruct the flight path that this person took, assuming each ticket was used exactly once.

If there are multiple valid flight paths, return the lexicographically smallest one.

For example, the itinerary ["JFK", "SEA"] has a smaller lexical order than ["JFK", "SFO"].
You may assume all the tickets form at least one valid flight path.

```python
# Reconstruct the itinerary in lexicographical (alphabetical) order. 
# build adjacency list (hint: reverse sorted order to force the lexicographical ordeR)
# initialise a stack with the starting node "JFK"
# initialise an empty itinerary list
# while stack:
# pop the neighbours of the node in the graph at the top of the stack to the stack using a while loop. 
# pop and append the node at the top of the stack to the itinerary (we are done with it)
# return a reversed itinerary. 
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = defaultdict(list)

        for src, dest in sorted(tickets, reverse=True):
            graph[src].append(dest)
        
        stack = ["JFK"] # starting node
        itinerary = []

        while stack:
            # While the node at the top of the stack has neighbors (destination airports)
            while graph[stack[-1]]:
                stack.append(graph[stack[-1]].pop()) # pop a neighbor from the node at the top of the stack and append it to the stack. 
            itinerary.append(stack.pop())
        
        return list(reversed(itinerary))
```

### Key Concepts:
- To visit the nodes in lexicographical (alphabetical) order, we need to reverse sort the tickets array
- Why Reverse?
    - When you build an adjacency list, you store the destinations for each source airport.
    - During traversal, we pop destinations from the adjacency list. 
    - Popping from a list removes the last element, so to geth the smallest (lexicographically first) destination first, you need to store the destinations in reverse sorted order. 
- Why reverse at the end?
    - This algorithm is a form of Hierholzer's algorithm for finding an Eularian path. 
    - The path is constructed in reverse order because you append nodes to the itinerary only after you have visited all their neighbors. 
    So, the itinerary list you build is actually the reverse of the final path you want. 
    To get the correct order, you reverse the itinerary before returning it. 

### Time and Space:
- Time:
1. Building the graph:
    - Sorting the tickets takes O(n log n)
    - Inserting each into the adjacency list takes O(n)
2. DFS traversal
    - Each edge is visited exactly once when popped from the adjacency list
    - each node (airport) is pushed and popped from the stack at most once per edge
    - So the DFS travesal takes O(n)
Total: O(n log n)
Space: O(n)

## Min Cost to Connect Points
You are given a 2-D integer array points, where points[i] = [xi, yi]. Each points[i] represents a distinct point on a 2-D plane.

The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between the two points, i.e. |xi - xj| + |yi - yj|.

Return the minimum cost to connect all points together, such that there exists exactly one path between each pair of points.

```python
# Return the minimum cost to connect all points together. 
# Prim's Algorithm
# Setup: init n, total_cost, seen set, and min_heap with a tuple (0,0)
# while the seen set length is less than n
# unpack distance and index from heap
# if in seen continue
# add to seen
# increment cost with distance
# unpack i coords from points array using index
# loop through j in n, if not in seen, unpack j coords
# calculate the distance between the i and j coords and push onto the heap. 
# return total cost. 
import heapq
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        total_cost = 0
        visited = set()
        min_heap = [(0,0)]

        while len(visited) < n:
            # unpack 
            dist, i = heapq.heappop(min_heap)
            if i in visited:
                continue
            
            visited.add(i)
            total_cost += dist

            xi, yi = points[i]

            for j in range(n):
                if j not in visited:
                    xj, yj = points[j]
                    nei_dist = abs(xi-xj) + abs(yi-yj)
                    heapq.heappush(min_heap, (nei_dist, j))
        
        return total_cost
```

### Key Concepts
- We can solve this using Prim's algorithm: a greedy method used to find the minimum spanning tree of a connected, weighted graph
- By starting from an arbitrary vertex and repeatedly adding the smallest edge that connects a vertex in the tree to a vertex outside it. 
- This process continues until all verices are included, ensuring the total edge weight is minimized. 
- In this solution, we use a min heap to keep track of the smallest "edge" in the heap. When we pop this edge, we check if we have already used that point, if not then we set it to visited. 
- For a given point popped form the heap, after it has been set to visited, we add all of the edge distances from that point to every other point to the heap.
- This means that any of those edges may be popped from the heap at some point and used, if that edge turned out to be the smallest distance available. 

### Time and Space:
- Time: O(n^2 log(n))
- Space: O(n^2)