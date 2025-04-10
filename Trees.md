# Trees

## Invert Binary Tree
You are given the root of a binary tree root. Invert the binary tree and return its root.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # if not root return none
        if not root: return None

        # swap root.left and root.right
        root.left, root.right = root.right, root.left

        # call invertTree left
        self.invertTree(root.left)
        # call invertTree right
        self.invertTree(root.right)

        return root
```

### Key Concepts
- This solution uses Depth First Search and Recursion to reverse the tree
- The base case is if not root, which causes the call stack to unravel. 
- every time we go down the tree, we reverse the nodes and go down the left and right node recursively until we hit the base case.


## Maximum Depth of Binary Tree

Given the root of a binary tree, return its depth.

The depth of a binary tree is defined as the number of nodes along the longest path from the root node down to the farthest leaf node.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0

        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)

        return 1 + max(left, right)
```

### Key Concepts
- Use a Depth First Search / Recursion to find the base case
- The basecase is "if not root", that means we can return 0 (the node does not exist)
- If a node exists then it reports back its own height of 1 + the max of the heights between its left and right leaf nodes. 

## Diameter of Binary Tree
The diameter of a binary tree is defined as the length of the longest path between any two nodes within the tree. The path does not necessarily have to pass through the root.

The length of a path between two nodes in a binary tree is the number of edges between the nodes.

Given the root of a binary tree root, return the diameter of the tree.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # Use Global variable to track largest diamater
        # We can use DFS to find the diameter of each node. 
        # Diameter of a node = height of left + height of right node
        # helper function to get the height of the node
        largest_diameter = 0

        def height(root):
            nonlocal largest_diameter
            if not root: return 0

            left_height = height(root.left)
            right_height = height(root.right)
            diameter = left_height + right_height
            largest_diameter = max(largest_diameter, diameter)

            return 1 + max(left_height, right_height)
        
        height(root)
        return largest_diameter
```

### Time and Space Complexity
- Time: O(n)
- Space: O(h) where h is the height of the tree

### Key Concepts
- We can use DFS to find the diameter of each node. We want to return the largest possible diameter, so we start at the deepest node using depth first search until we have calculated the diameter of each node
- Diameter of a node can be calculated using height of the left node and height of the right node
- Height of a given node is given by the formula 1 + max(left_height, right_height)
- We need to use a helper function to recursively calculate the height of the left and right leaf nodes. 
- We need a global variable to keep track of the largest_diameter. 
- When using a global variable in a recursive function like this, we need to intialise it as "nonlocal"


## Balanced Binary Tree
Given a binary tree, return true if it is height-balanced and false otherwise.

A height-balanced binary tree is defined as a binary tree in which the left and right subtrees of every node differ in height by no more than 1.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # We want to store a global balanced variable and set it to True (use an array to store it because python)
        # we want to use a helper function to perform DFS on the tree to determine if each node in the tree is balanced. 
        # Balanced means that the left height and right height have a difference of no more than 1. 
        # we call the helper function
        # return the balanced variable. 
        balanced = [True]

        def height(root):
            if not root: return 0

            left_height = height(root.left)
            right_height = height(root.right)

            if abs(left_height - right_height) > 1:
                balanced[0] = False
                return 0

            return 1 + max(left_height, right_height)

        height(root)
        return balanced[0]
```

### Key Concepts
- Use a global boolean to keep track if we find an unbalanced node in the tree. 
- We want to use a helper function to perform DFS on the tree to determine if each node in the tree is balanced. 
- Balanced means the left height and right height have a difference of no more than 1. 
- We can use the absolute function to help us calculate the difference between two heights. 
- Absolute value means distance from 0. Example: 10-13 = -3, this gets converted to 3 because its distance from 0 is 3.
- We can use the in-built abs function to find the absolute value. 

### Time and Space Complexity
- Time: O(n) where n is the number of nodes in the tree, we must visit every node once. 
- Space: O(h) Because its a DFS, where h is the height of the tree. 

## Same Binary Tree
Given the roots of two binary trees p and q, return true if the trees are equivalent, otherwise return false.

Two binary trees are considered equivalent if they share the exact same structure and the nodes have the same values.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # use a helper function to perform DFS on both trees at the same time. 
            # true if both nodes are none
            # false if either node is none but the other isn't
            # false if nodes are not the same value
            # return the result of the helper function of each tree, left and right
        # return the helper function of the input trees. 
        def balanced(p, q):
            if not p and not q:
                return True
            
            if (p and not q) or (q and not p):
                return False

            if p.val != q.val:
                return False

            return balanced(p.left, q.left) and balanced(p.right, q.right)

        return balanced(p, q)
```

### Key Concepts
- We can use DFS to check if each node is balanced. 
- Our base case is that both nodes are None. 
- If one node exists, and another doesn't, its not balanced
- If either node has a different value, its not balanced

### Time and Space
- Time: O(n + m)
- Space: O(h_p + h_q) where h_p and h_q are the height of each tree respectively.

## Subtree of Another Tree
Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:   
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # need to define a sameTree helper function
        # need to define a has_subtree helper function
        # has_subtree checks for base case (if not Root)
        # return true if sameTree between root and subRoot is true
        # recursively call has_subtree on the left and right nodes

        def sameTree(p, q):
            if not p and not q: return True # base case
            if (q and not p) or (p and not q): return False 
            if p.val != q.val: return False

            return sameTree(p.left, q.left) and sameTree(p.right, q.right)
        
        def has_subtree(root):
            if not root: return False

            if sameTree(root, subRoot):
                return True
            
            return has_subtree(root.left) or has_subtree(root.right)
        
        return has_subtree(root)
```

### Key Concepts
- We use a helper function (sameTree) to check if each node is the same as the subRoot tree
- Use DFS to call the sameTree function on each node. 


## Lowest Common Ancestor in Binary Search Tree
Given a binary search tree (BST) where all node values are unique, and two nodes from the tree p and q, return the lowest common ancestor (LCA) of the two nodes.

The lowest common ancestor between two nodes p and q is the lowest node in a tree T such that both p and q as descendants. The ancestor is allowed to be a descendant of itself.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # Use global variable to track LCA, set it to Root (root will always be a common ancestor)
        # Use a DFS helper function to traverse the tree, set the LCA to the current node.
        # conditions:
        # if root is p or q return
        # if root.val is less than p and q search right
        # if root.val is more the p and q search left
        # else return (we are between the p and q therefore we have found the LCA)
        lca = [root]
        def search(node):
            if not node: return

            lca[0] = node
            if node is p or node is q:
                return # found LCA
            elif node.val < p.val and node.val < q.val:
                search(node.right)
            elif node.val > p.val and node.val > q.val:
                search(node.left)
            else:
                return # found LCA
        
        search(root)
        return lca[0]
```

### Key Concepts
- We can use DFS to traverse the tree
- Use a global variable to keep track of the lowest common ancestor
- Update the LCA every time we visit a new node
- Binary Search Tree properties means that the values to the left will be less than the values to the right so we can perform a binary search.
- if both p and q values are less than node value we want to search left
- if both p and q values are greater than node value we want to search right
- if the node is between p and q, we have found the LCA

## Binary Tree Level Order Traversal
Given a binary tree root, return the level order traversal of it as a nested list, where each sublist contains the values of nodes at a particular level in the tree, from left to right.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # Use BFS to traverse tree
        # return if root none
        # initialise queue, append root, initialise answer list
        # loop while queue exists
            # initialise level list, set n to the length of the queue
            # inner for loop of length n
                # grab node from queue using popleft
                # append the node value to the level list
                # append the left and right nodes to the queue if they exist
            # append level to answer list
        #return answer list

        if root is None: return []

        queue = deque()
        queue.append(root)
        answer = []

        while queue:
            level = []
            n = len(queue)
            for i in range(n):
                node = queue.popleft()
                level.append(node.val)

                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)

            answer.append(level)
        return answer
```

### Key Concepts
- We can use Breadth First Search to traverse the tree in level order.
- Every time we visit a node we store the left and right "children" nodes in the queue
- We popleft from the queue to process that node
- The loop finishes when we exhaust the queue. 

## Binary Tree Right Side View
You are given the root of a binary tree. Return only the values of the nodes that are visible from the right side of the tree, ordered from top to bottom.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        # Check if Root exists
        # Level Order Traversal using BFS
        # Define a queue with the value of root, define a result list
        # loop while queue
            # store length of q in variable (level length)
            # loop through length of q
                # popleft the node from the queue
                # if index is equal to level length minus one. This means we have the rightmost node in the level. Append node.val to result
                # if node left and right, append those nodes to the queue
        # return result
        if not root: return []
        queue = deque([root])
        res = []

        while queue:
            level_len = len(queue)
            for i in range(level_len):
                node = queue.popleft()
                if i == level_len - 1: res.append(node.val) # This gets us the right most node in the level.
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
        
        return res
```

### Key Concepts
- We use the same algorithm as the Level Order Traversal with Breadth First Search
- We can use a queue to force the level traversal.
- We store the length of the queue before the for loop because the for loop itself is going to add new nodes to the queue. So we want to ensure we only loop for the nodes in the queue that are on the level. 
- Key basecase for each level, if the index of the inner for loop is equal to the length of the level - 1, then we are at the rightmost node for that level

## Count Good Nodes in Binary Tree
Within a binary tree, a node x is considered good if the path from the root of the tree to the node x contains no nodes with a value greater than the value of node x

Given the root of a binary tree root, return the number of good nodes within the tree.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        # Can solve using Iterative DFS using a stack
        # keep a good_nodes counter, initialise a stack with a tuple root, and float('-inf')
        # loop while stack
        # unpack the tuple from the stack using pop
        # if the largest value is less than or equal to the node, this is a good node, increment good_nodes counter
            # set largest to the max of largest and node.val
            # if node.right and node.left we want to stack.append the tuple of the node and largest. 
        good_nodes = 0
        stack = [(root, float('-inf'))]

        while stack:
            node, largest = stack.pop()

            if largest <= node.val:
                good_nodes +=1
            
            largest = max(largest, node.val)

            if node.right: stack.append((node.right, largest))
            if node.left: stack.append((node.left, largest))

        return good_nodes

            
        #return the good nodes
```

### Key Concepts
- We can use Iterative DFS to solve this question
- Maintain a stack which stores a tuple for each node containing the node itself and the largest value up until that node was added to the stacck
- When we unpack the node and largest value from the stack, we can compare the two values to determine if its a good node. 
- We then calculate the new largest
- Then add the left or right nodes to the stack with the new largest value if they exist. 
- return the final good_nodes count. 

## Time and Space Complexity
- Time: O(n), since we traverse through every node in the tree using DFS
- Space: O(n), in iterative DFS we maintain a stack in memory which can at worst contain up to n nodes. 


## Valid Binary Search Tree
Given the root of a binary tree, return true if it is a valid binary search tree, otherwise return false.

A valid binary search tree satisfies the following constraints:

The left subtree of every node contains only nodes with keys less than the node's key.
The right subtree of every node contains only nodes with keys greater than the node's key.
Both the left and right subtrees are also binary search trees.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# Valid BST: left subtree contains only nodes with keys less than, right subtree contains only nodes with keys greater than
# Create a helper function with 3 parameters, the node, the minimum, the maximum
# the base case is if not node
# if the node value is outside of the min max bounds return False
# we want to return the helper function recursively for the left and the right. 
# outside the helper function we simply return the helper function with min max infinity


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def is_valid(node, minimum, maximum):
            if not node: return True

            if node.val <= minimum or node.val >= maximum:
                return False
            
            return is_valid(node.left, minimum, node.val) and is_valid(node.right, node.val, maximum)
        
        return is_valid(root, float('-inf'), float('inf'))

```

### Key Concepts
- We can use recursive DFS to solve this problem.
- We can use a helper function to traverse down the left or the right of the tree
- If we traverse down the left, we always want to check that the left node is less than the current node. So we simply pass the current node value as the maximum
- Same with down the right of the tree, we want to check if each node is strictly bigger than the previous, so we set the minimum to the current node before we pass it down. 
- The base case is if we find a null node, then we return True which will begin to unravel the callstack. 

### Time and Space Complexity
- Time: O(n), because we are visiting every node once using DFS
- Space: O(h), where h is the height of the tree. (DFS needs to store at least h nodes in the callstack)

## Kth Smallest Integer in BST
Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) in the tree.

A binary search tree satisfies the following constraints:

The left subtree of every node contains only nodes with keys less than the node's key.
The right subtree of every node contains only nodes with keys greater than the node's key.
Both the left and right subtrees are also binary search trees.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# Inorder traversal using Recursive DFS
# initialise 2 global variables count and answer. 
# define dfs helper function
# base case simply returns
# call dfs on the left node
# check if count is 1: set answer to node value
# decrement count
# if count is above 0: call dfs on right node
# call dfs on the root and return answer. 

class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        count = [k]
        answer = [0]

        def dfs(node):
            if not node: return

            dfs(node.left)

            if count[0] == 1:
                answer[0] = node.val
            
            count[0] -= 1
            if count[0] > 0:
                dfs(node.right)
        
        dfs(root)
        return answer[0]
```

### Key Concepts
- Can be solved using Inorder Traversal using Recursive DFS
- Since we want to return the kth smallest element, we want to traverse through the BST k times. 
- If we decrement the k value every time we traverse a node, this will help us identify the kth smallest. 
- Because of the BST properties, once we reach our first base case, we will begin processing each node and decrementing the counter.


## Construct Binary Tree from Preorder and Inorder Traversal
You are given two integer arrays preorder and inorder.

preorder is the preorder traversal of a binary tree
inorder is the inorder traversal of the same tree
Both arrays are of the same size and consist of unique values.
Rebuild the binary tree from the preorder and inorder traversals and return its root.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # Base Case
        if not preorder or not inorder:
            return None
        
        root_val = preorder[0]
        root = TreeNode(root_val)

        root_index = inorder.index(root_val) # This finds the index of the current root value in the inorder list

        # Recursively build the left tree
        root.left = self.buildTree(preorder[1:root_index+1], inorder[:root_index])
        # Recursively build the right tree
        root.right = self.buildTree(preorder[root_index+1:], inorder[root_index+1:])
        return root
```

### Key Concepts
- In preorder traversal: root --> left subtree --> right subtree
- In inorder traversal: left subtree --> root --> right subtree
- First element in preorder is the root
- We can then split the inorder list using the index of the root that we found
- For Left Subtree: elements before root in inorder, elements after root for preorder
- For Right Subtree: elements after root in inorder, elements before root for preorder

- Creating a node: node = TreeNode(value)
- Array Slicing: array[start:stop:step]
- start is inclusive (we include the element at this index)
- stop is exclusive (we exclude the element at this index)

Example:
- preorder = [1,2,3,4], inorder [2,1,3,4]
1. The root is the first element of preorder, which is 1. [(1),2,3,4]
2. Find the position/index of root (1) in the inorder array: it's at index 1. [2,(1),3,4]
So root_index = 1
Left Subtree: 
- preorder = [1,2,3,4], inorder = [2,1,3,4]
- preorder[1:root_index+1] = preorder[1:1+1] = preorder[1:2] = [2] "include element at index 1, but not at index 2"
- inorder[:root_index] = [2]
Right Subtree: 
- preorder[root_index+1:] = preorder[1+1:] = preorder[2:] = [3,4] "start from index + 1, until the end of the list"
- inorder[root_index+1:] = [3,4] "every element after the root index"
So we end up with 
```
root.left = self.buildTree([2],[2])
root.right = self.buildTree([3,4],[3,4])
```

## Binary Tree Maximum Path Sum
Given the root of a non-empty binary tree, return the maximum path sum of any non-empty path.

A path in a binary tree is a sequence of nodes where each pair of adjacent nodes has an edge connecting them. A node can not appear in the sequence more than once. The path does not necessarily need to include the root.

The path sum of a path is the sum of the node's values in the path.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        if not root: return 0

        self.max_path_sum = root.val
        self.dfs(root)

        return self.max_path_sum

    def dfs(self, node):
        if not node: return 0

        # If leaf node, return max of node val or global max
        if not node.left and not node.right:
            self.max_path_sum = max(node.val, self.max_path_sum)
            return node.val
        
        # Call DFS on the left and right nodes
        l_path_sum = self.dfs(node.left)
        r_path_sum = self.dfs(node.right)

        self.max_path_sum = max(
            self.max_path_sum,
            node.val,
            node.val + l_path_sum,
            node.val + r_path_sum,
            node.val + l_path_sum + r_path_sum
        )

        return max(
            node.val,
            node.val + l_path_sum,
            node.val + r_path_sum,
            0
        )

# Time: O(n), since we are visiting every node atleast once using DFS
# Space: O(h), where h is the height of the tree. 
```

### Key Concepts:
- Postorder Traversal with DFS
- maintain a global variable to track the maximum path sum
- call a DFS helper function to compute the max path sum from the root
- Inside helper function:
- - Check if not node, return 0 (first base case)
- - if leaf node, second base case, (no left or right nodes): 
- we want to set the max path sum to the max of the global variable compared to the current node value. then early return
- calculate left path sum and right path sum on left and right subtrees using helper function
- we want to then set the max global variable to the max of:
- - the global max
- - the node value
- - node + left and right path sums
- - node + left
- - node + right
= return the max of the node,  node + left, node + right or 0. 


## Serialize and Deserialize Binary Tree
Implement an algorithm to serialize and deserialize a binary tree.

Serialization is the process of converting an in-memory structure into a sequence of bits so that it can be stored or sent across a network to be reconstructed later in another computer environment.

You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure. There is no additional restriction on how your serialization/deserialization algorithm should work.

Note: The input/output format in the examples is the same as how NeetCode serializes a binary tree. You do not necessarily need to follow this format.

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Codec:
    
    # Encodes a tree to a single string.
    def serialize(self, root: Optional[TreeNode]) -> str:
        res = []

        def dfs(node):
            if not node: 
                res.append("N")
                return
            
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        return ",".join(res)
        
    # Decodes your encoded data to tree.
    def deserialize(self, data: str) -> Optional[TreeNode]:
        vals = data.split(",")
        self.i = 0

        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None # Basecase
            
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs() # This will incrememnt the index until we are ready to process the node.right
            node.right = dfs()

            return node
        
        return dfs()
```

### Key Concepts:
- We can use DFS or BFS to solve this, but DFS is less code. 
- We are using Preorder traversal to build the serialize string: "Node --> Left --> Right" 
- Our base case is if we have Null left and Null right nodes. This identifies a Leaf Node. 
- Every node we visit during the preorder traversal gets appended to the list. 
- In Deserialize, we can rebuild the tree in a similar way
- The basecase is that the current value in the list is the special character "N". This returns None because we don't need to attach a None node to the tree
- If we pass the base case, it means we are processing an actual value. We simply use the TreeNode class to create a new node. 
- We then recursively build its left and right children. This works because the index keeps track of where we are in the input list, and when we are finished processing the left children, the index will be at the first node for the right children. 
- We need to make sure we return the node at the end of the DFS function
- We can simply just return the dfs helper function which will return the root node of the reconstructed tree. 

### Time and Space Complexity
- Time: O(n), because its a DFS for both serialize and deserialize
- Space: O(n), We need to store every node in the result list for serialise, and we need to store the callstack in memory aswell which would be the height of the tree. This would be O(n+h) but it can be simplified to O(n).