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
