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