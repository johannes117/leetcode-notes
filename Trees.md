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

### Key Concepts
- We can use DFS to find the diameter of each node. We want to return the largest possible diameter, so we start at the deepest node using depth first search until we have calculated the diameter of each node
- Diameter of a node can be calculated using height of the left node and height of the right node
- Height of a given node is given by the formula 1 + max(left_height, right_height)
- We need to use a helper function to recursively calculate the height of the left and right leaf nodes. 
- We need a global variable to keep track of the largest_diameter. 
- When using a global variable in a recursive function like this, we need to intialise it as "nonlocal"
