# Stacks

# Key Data Structure
A stack is a LIFO (Last In First Out) data structure.
Python Implementation:
```python
stack = []
stack.append(1) # [1]
stack.append(2) # [1, 2]
stack.pop() # [1]
stack[-1] # 1
len(stack) == 0 # False
```

## Valid Parentheses
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

1. Open brackets are closed by the same type of brackets.
2. Open brackets are closed in the correct order.
3. Every close bracket has a corresponding open bracket.

## Key Data Structure
A stack is a LIFO (Last In First Out) data structure.

## Solution
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        closeToOpen = {")": "(", "}": "{", "]": "["}

        for c in s:
            if c in closeToOpen:
                if stack and stack[-1] == closeToOpen[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)
        return True
```

### Key Concepts
- A stack is used to store the open brackets
- A dictionary is used to store the close brackets and their corresponding open brackets
- The stack is popped if the top of the stack is the corresponding open bracket
- The stack is returned true if it is empty, otherwise false

## Evaluate Reverse Polish Notation
You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

Note that:

- Valid operators are '+', '-', '*', and '/'.
- Each operand may be an integer or another expression.
- The division between two integers should truncate toward zero.
- The given RPN expression is always valid. That means the expression would always evaluate to a result, and there will not be any division by zero operation.

## Solution
```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        # hashSet to store operator
        operatorSet = {"+", "-", "/", "*"}
        stack = []

        # loop through value in list
        for c in tokens:
            if c not in operatorSet:
                stack.append(int(c))
            else:
                leftValue = stack.pop()
                rightValue = stack.pop()
                if c == "+":
                    newValue = leftValue + rightValue
                elif c == "-":
                    newValue = rightValue - leftValue
                elif c == "/":
                    newValue = int(rightValue /  leftValue)
                elif c == "*":
                    newValue = leftValue * rightValue
                stack.append(newValue)
        return stack[-1]
```

### Key Concepts
- A stack is used to store the operands
- A hashSet is used to store the operators
- The stack is popped if the top of the stack is an operator
- The stack is returned the last value in the stack
