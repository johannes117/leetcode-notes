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


## Generate Parentheses

You are given an integer n. Return all well-formed parentheses strings that you can generate with n pairs of parentheses.

Example 1:

Input: n = 1

Output: ["()"]
Example 2:

Input: n = 3

Output: ["((()))","(()())","(())()","()(())","()()()"]
You may return the answer in any order.

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        stack = []
        res = []

        def backtrack(openN, closedN):
            # Basecase: if openN and closedN equal n, then append stack to result. res.append("".join(stack))
            if openN == closedN == n:
                res.append("".join(stack))
                return

            # if openN < n: This will recursively traverse down the left decision tree first
            if openN < n:
                # append a open parenthesis to the stack
                stack.append("(")
                # recursively call backtracking function (once it hits the basecase (openN == closedN == n)it will return to this point)
                backtrack(openN + 1, closedN) # we increment the openN value by 1 so that we calculate the next level down in the left tree
                # pop from the stack (This essentially the undo step of the backtracking algo)
                stack.pop()
            # same thing but for the right decision tree. 
            if closedN < openN: # We ensure that we only add a close parenthesis if there are more open parentheses than close parentheses
                stack.append(")")
                backtrack(openN, closedN + 1) # increment closedN value by 1 to traverse down the right of the decision tree.
                stack.pop()

        backtrack(0, 0) # pass in 0 index for closed and open 
        return res
```

### Key Concepts
- A stack is used to store the parentheses
- A list is used to store the result
- we use backtracking to generate all the possible combinations of parentheses
- backtracking is useful when we need to generate all possible combinations of a set of values

## Key Algorithm
- Backtracking: A general algorithm for finding all (or some) solutions to some computational problems, notably constraint satisfaction problems, that incrementally builds candidates to the solutions, and abandons a candidate ("backtracks") as soon as it determines that the candidate cannot possibly be completed to a valid solution.
- Make a decision, explore it, and then undo the decision.

## Daily Temperatures
You are given an array of integers temperatures where temperatures[i] represents the daily temperatures on the ith day.

Return an array result where result[i] is the number of days after the ith day before a warmer temperature appears on a future day. If there is no day in the future where a warmer temperature will appear for the ith day, set result[i] to 0 instead.

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # initialise res list with all 0's with the length of temperatures list
        res = [0] * len(temperatures)
        # initialise stack pair: [temp, index]
        stack = []

        # enumerate through temperatures and get the index and value/temp in temp list
        for i, t in enumerate(temperatures):
            # while stack exists and temperature value is greater than the temp value at the top of the stack:
            while stack and t > stack[-1][0]: # stack[-1][0] is the temperature at the top of the stack in python
                # pop from stack and unpack index and value
                poppedTemp, poppedIndex = stack.pop()
                # insert the distance from current index and popped index into result list at the popped index
                res[poppedIndex] = (i - poppedIndex)
            # append current temp and index to stack
            stack.append([t, i])
        # return result list
        return res
```

### Key Concepts
- A Monotonic decreasing stack is used to store the temperatures and their corresponding index pairs
- The stack is popped if the temperature at the top of the stack is less than the current temperature
- The result list is updated with the distance between the current index and the index at the top of the stack
- The stack is returned with the result list

### Monotonic Stack
- A monotonic stack is a stack where the elements are in a sorted order
- A monotonic decreasing stack is a stack where the elements are in a decreasing order
- A monotonic increasing stack is a stack where the elements are in an increasing order

## Car Fleet
There are n cars traveling to the same destination on a one-lane highway.

You are given two arrays of integers position and speed, both of length n.

position[i] is the position of the ith car (in miles)
speed[i] is the speed of the ith car (in miles per hour)
The destination is at position target miles.

A car can not pass another car ahead of it. It can only catch up to another car and then drive at the same speed as the car ahead of it.

A car fleet is a non-empty set of cars driving at the same position and same speed. A single car is also considered a car fleet.

If a car catches up to a car fleet the moment the fleet reaches the destination, then the car is considered to be part of the fleet.

Return the number of different car fleets that will arrive at the destination.

```python
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        # convert position and speed lists into a pair list using zip function
        pair = [(p, s) for p, s in zip(position, speed)]
        # sort pair list in reverse
        pair.sort(reverse=True)
        # initialise stack
        stack = []

        # for position and speed in reversed pair list
        for p, s in pair:
            # append the time of arrival to the stack: (target - position) / speed
            stack.append((target - p) / s)
            # if length of stack is greater than or equal to 2 and top of stack is less than or equal to the 2nd from the top of stack
            # We check stack >= 2 otherwise stack[-2] will give us an index error if there is only 1 value in the stack
            # when the current car arrival time: stack[-1] is less than or equal to stack[-2] (car in front), 
            # That means the current car will intercept the car in front forming a fleet.
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                # pop from stack
                stack.pop()
        # return length of stack. 
        return len(stack)
```

### Key Concepts
- A stack is used to store the time of arrival of each car
- The stack is popped if the time of arrival of the current car is less than or equal to the time of arrival of the car in front
- The stack is returned with the number of different car fleets that will arrive at the destination
- A fleet can be tracked using the time of arrival of the car at the front of the fleet. Therefore we can pop cars that are part of the fleet from the stack.

### Key Techniques
- Zipping two lists together: 
```python
pair = [(p, s) for p, s in zip(position, speed)]
```
- Sorting a list in reverse order:
```python
pair.sort(reverse=True)
```

### Time Complexity
- O(n log n) for sorting the pair list
- O(n) for iterating through the pair list
- O(1) for popping from the stack
- Therefore, the time complexity is O(n log n)
