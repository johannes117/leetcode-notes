# Linked Lists

## Reverse a Linked Lists
Given the beginning of a singly linked list head, reverse the list, and return the new beginning of the list.


### Key Concepts
- Use Two Pointers: Current and Previous
- curr = head
- prev = null
- use Temporary Variable to store the next node
- t = curr.next
- curr.next = prev
- prev = curr
- curr = t
- return prev
- performing curr.next = prev reverses the link!

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # initialise current and previous pointers
        curr = head
        prev = None

        # while current pointer is not none
        while curr:
            # set temp value to the next value from current 
            t = curr.next
            # Reverse the link: Set current.next to the previous value
            curr.next = prev # reversing the link
            # set previous to the current value
            prev = curr
            # set current to the temp value
            curr = t
        
        # return previous
        return prev
```


## Merge Two Sorted Lists
You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists into one sorted linked list and return the head of the new sorted linked list.

The new list should be made up of nodes from list1 and list2.

### Solution
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # Create a dummy node to handle edge cases and simplify the merging process
        # The dummy node serves as the starting point of our merged list
        d = ListNode()
        curr = d

        # Continue while both lists have nodes to compare
        while list1 and list2:
            # Compare values from both lists and link the smaller value to our merged list
            if list1.val < list2.val:
                # Link current node to the smaller value (list1)
                curr.next = list1
                # Move curr pointer to the newly added node
                curr = list1
                # Advance list1 to its next node
                list1 = list1.next
            else:
                # Link current node to list2 (when list2 val is smaller or equal)
                curr.next = list2
                curr = list2
                list2 = list2.next

        # At this point, at least one list is exhausted
        # Append the remaining nodes from either list1 or list2
        # Since the remaining list is already sorted, we can link it directly
        if list1:
            curr.next = list1
        else:
            curr.next = list2

        # Return the head of the merged list (skip the dummy node)
        return d.next
```

### Key Concepts
- We need to create a dummy node to handle edge cases. The Dummy node also acts as a bit of a starting point
- We set the current pointer to the dummy node at the beginning
- We use a while loop until one of the two lists are exhausted
- Since the lists are sorted, we want to take the lower value of each lists current head node. 
- if a list.value is lower, we link the current note to the smaller values list, and move the pointer to the newly added node. 
- We then advance the list to its next node. 
- Once we are finished with the while loop, we append whichever list still has values to the end of our merged list, and return dummy.next. 

## Linked List Cycle Detection
Given the beginning of a linked list head, return true if there is a cycle in the linked list. Otherwise, return false.

There is a cycle in a linked list if at least one node in the list can be visited again by following the next pointer.

Internally, index determines the index of the beginning of the cycle, if it exists. The tail node of the list will set it's next pointer to the index-th node. If index = -1, then the tail node points to null and no cycle exists.

Note: index is not given to you as a parameter.

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # Create a dummy node to handle edge cases
        d = ListNode()
        d.next = head
        # Set both slow and fast pointers to the dummy node
        s = f = d

        while f and f.next:
            f = f.next.next
            s = s.next

            if s is f:
                return True
        
        return False
```

### Key Concepts
- We create a dummy node to handle edge cases.
- We set both slow and fast pointers to the dummy node.
- We use a while loop to move the fast pointer two steps at a time and the slow pointer one step at a time.
- If the slow and fast pointers meet, we return True.
- If the fast pointer reaches the end of the list, we return False.

## Reorder Linked List
You are given the head of a singly linked-list.

The positions of a linked list of length = 7 for example, can intially be represented as:

[0, 1, 2, 3, 4, 5, 6]

Reorder the nodes of the linked list to be in the following order:

[0, 6, 1, 5, 2, 4, 3]

Notice that in the general case for a list of length = n the nodes are reordered to be in the following order:

[0, n-1, 1, n-2, 2, n-3, ...]

You may not modify the values in the list's nodes, but instead you must reorder the nodes themselves.
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        # Find Middle 
        slow, fast = head, head.next
        # while loop until fast pointer hits the last node or none (while fast and fast.next)
        while fast and fast.next:
            # set the slow pointer to slow next
            slow = slow.next
            # set the fast pointer to fast next next
            fast = fast.next.next

        # second half list starts at the new slow.next value
        second = slow.next
        # prev = slow.next = None # This breaks the link of the two lists, and adds a None value to the end of the first list
        prev = slow.next = None

        # Reverse Second Half
        # while second half of list exists
        while second:
            # use temp value to store second next
            temp = second.next
            # second next = prev # This reverses the link
            second.next = prev
            # set prev to second
            prev = second
            # set second to temp
            second = temp

        # first list is at head
        first = head
        # second list is at prev
        second = prev

        # Merge two lists
        # while second
        while second:
            # create a temp variable for first and second nexts
            temp1, temp2 = first.next, second.next
            # set first next to second and second next to temp 1
            first.next = second
            second.next = temp1 # (This sets the value to what first.next was before we broke the link)
            # set first and second to the respective temp values
            first, second = temp1, temp2
```

### Key Concepts
- We find the middle of the list using the slow and fast pointers.
- We split the list into two halves.
- We reverse the second half of the list.
- We merge the two lists.
- We return the merged list.

Using slow and fast pointers to find the middle of a list:
Since the fast pointer moves twice as fast as the slow pointer, when the fast pointer reaches the end of the list, the slow pointer will be at the middle of the list.
We can use this as a split point to split the list into two halves.
We want to break the link of the second half of the list, so we set prev = slow.next = None. This breaks the link of the two lists, and adds a None value to the end of the first list.
Then we need to reverse the second half of the list. We do this by setting the second.next = prev, and then setting prev = second, and second = temp. This reverses the link of the second half of the list.
We then merge the two lists by setting first.next = second, and second.next = temp1. This sets the value to what first.next was before we broke the link.
We then set first and second to the respective temp values.
Head will then be the new head of the merged list.

## Remove Node From End of Linked List
You are given the beginning of a linked list head, and an integer n.

Remove the nth node from the end of the list and return the beginning of the list.

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # initialise dummy node
        dummy = ListNode()
        dummy.next = head
        # initialise behind and head pointers to dummy
        ahead = behind = dummy

        # this loop places the ahead pointer at n+1 from the head of the list. This is useful so we can place the behind pointer later
        for _ in range(n+1):
            # set ahead pointer to ahead.next
            ahead = ahead.next
        
        # we move the ahead and behind pointers together until the ahead pointer reaches the end of the list.
        # since the ahead pointer is exactly n+1 jumps away from the behind pointer: the behind pointer will be at the n - 1 node.
        while ahead:
            # set behind to behind.next
            behind = behind.next
            # set ahead to ahead.next
            ahead = ahead.next

        # this code simply breaks the link for the target node and "skips" it. 
        behind.next = behind.next.next

        # this will return the new head of the list. 
        return dummy.next
```

### Key Concepts
- We create a dummy node to handle edge cases.
- We set the behind and ahead pointers to the dummy node.
- We use a for loop to place the ahead pointer at n+1 from the head of the list. This is useful so we can place the behind pointer later.
- We use a while loop to move the ahead and behind pointers together until the ahead pointer reaches the end of the list. Since the ahead pointer is exactly n+1 jumps away from the behind pointer: the behind pointer will be at the n - 1 node.
- We then break the link for the target node which "skips" it.
- We return the new head of the list.

## Copy Linked List with Random Pointer
You are given the head of a linked list of length n. Unlike a singly linked list, each node contains an additional pointer random, which may point to any node in the list, or null.

Create a deep copy of the list.

The deep copy should consist of exactly n new nodes, each including:

The original value val of the copied node
A next pointer to the new node corresponding to the next pointer of the original node
A random pointer to the new node corresponding to the random pointer of the original node
Note: None of the pointers in the new list should point to nodes in the original list.

Return the head of the copied linked list.

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return None
        # use a dictionary to store old to new values
        old_to_new = {}
        curr = head

        # First loop to create the old_to_new hashmap
        while curr:
            node = Node(x=curr.val) # create a new node with the value of the current node
            old_to_new[curr] = node # store the new node in the hashmap
            curr = curr.next # move to the next node
        
        curr = head
        # second while loop
        while curr:
            new_node = old_to_new[curr] # get the new node from the hashmap
            new_node.next = old_to_new[curr.next] if curr.next else None # set the next pointer of the new node to the next pointer of the current node
            new_node.random = old_to_new[curr.random] if curr.random else None # set the random pointer of the new node to the random pointer of the current node
            curr = curr.next # move to the next node

        return old_to_new[head]
```

### Key Concepts
- We use a dictionary to store old to new values.
- We first loop through the list to create the old_to_new hashmap.
- We then loop through the list again to create the new nodes and set the next and random pointers.
- We return the new head of the list.

## Find Duplicate Integer in Linked List

Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = nums[0], nums[0]

        # First phase: find the intersection point
        while True:
            slow = nums[slow] # Move slow pointer one step
            fast = nums[nums[fast]] # Move fast pointer two steps
            if slow == fast: # If the two pointers meet, we have found the intersection point
                break
        
        slow2 = nums[0]
        while slow != slow2:
            slow = nums[slow] # Move slow pointer one step
            slow2 = nums[slow2] # Move slow2 pointer one step
        return slow
```

### Key Concepts
- We use the slow and fast pointers to find the duplicate number.
- We need to use Floyd's Tortoise and Hare (Cycle Detection) algorithm to find the duplicate number.
- We need to find the intersection of the two pointers.
- We then need to find the entrance to the cycle.
- Use a second slow pointer to find the entrance to the cycle.
- When the two slow pointers meet, we have found the duplicate number.
- We return the duplicate number.
