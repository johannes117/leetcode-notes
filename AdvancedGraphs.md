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