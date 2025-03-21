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