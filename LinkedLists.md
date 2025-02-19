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
