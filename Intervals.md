# Intervals

## Insert Interval
You are given an array of non-overlapping intervals intervals where intervals[i] = [start_i, end_i] represents the start and the end time of the ith interval. intervals is initially sorted in ascending order by start_i.

You are given another interval newInterval = [start, end].

Insert newInterval into intervals such that intervals is still sorted in ascending order by start_i and also intervals still does not have any overlapping intervals. You may merge the overlapping intervals if needed.

Return intervals after adding newInterval.

Note: Intervals are non-overlapping if they have no common point. For example, [1,2] and [3,4] are non-overlapping, but [1,2] and [2,3] are overlapping.

```python
# Insert Interval into list and return new list
# 3 Cases: current last number in current interval is less than the first number in the newInterval: append curr interval to result
# current first number in interval is greater than last number in new interval: append new interval + the rest of the intervals
# else: we have an overlap, we need to merge [min of the starts, max of the ends]
# after loop append the newInterval, return result. 
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        
        for i, interval in enumerate(intervals):
            # If interval less than and does not overlap
            if interval[1] < newInterval[0]:
                result.append(interval)
            # If interval greater than and does not overlap
            elif interval[0] > newInterval[1]:
                result.append(newInterval)
                return result + intervals[i:]
            else:
                newInterval = [min(interval[0], newInterval[0]),max(interval[1], newInterval[1])]
        
        result.append(newInterval)
        return result
```

### Key Concepts:
- We can solve this by iterating through the intervals list, and handling 3 cases. 
- Case 1: Current Interval does not overlap with New Interval, and its less than. Append to result
- Case 2: Current interval does not overlap with New interval, and its more than. Append newInterval and the rest of the intervals to the result and early return
- Case 3: Current interval is overlapping, set new interval to [min of mins, max of maxs]
- If we make it out of the loop that means our new interval does on the end of our result list


### Time and Space:
- Time: O(n)
- Space: O(n)

## Merge Intervals
Given an array of intervals where intervals[i] = [start_i, end_i], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

You may return the answer in any order.

Note: Intervals are non-overlapping if they have no common point. For example, [1, 2] and [3, 4] are non-overlapping, but [1, 2] and [2, 3] are overlapping.

```python
# Return an array of intervals after merging overlapping intervals
# Edgecase: if len less than or equal to 1 just return intervals
# Sort intervals using lambda function
# insert the first interval into the merged list
# loop through intervals starting from the second position. 
# if the first value in the current interval is less than or equal to the last element of the last interval in the merged list, 
# update the end time in the merged array to the max of the two intervals 
# else: append the current interval to the merged list
# return merged
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) <= 1:
            return intervals

        intervals.sort(key=lambda x: x[0]) # Sort by first element in interval
        merged = [intervals[0]]

        for current in intervals[1:]:
            # Check if overlapping
            if current[0] <= merged[-1][1]:
                merged[-1][1] = max(current[1], merged[-1][1])
            else:
                merged.append(current)
        
        return merged
```

### Key Concepts:
- We need to sort the intervals list so that its easier to work with.
- Lambda function basics: 
    - Syntax: lambda defines a nameless function, and then 'x' is the input parameter of the function (you can use any variable name for the input parameter)
    - lambda x: x[0] is the same as:
    ```python
    def get_last(x):
        return x[0]
    ```
    - So when we want to sort intervals based on a specific position we can use intervals.sort(key=lambda x: x[0])
- We insert the first interval into the output list to get us started. 
- We then loop through the remaining intervals
- Check if overlapping: if the current first value, is less than or equal to the last value in the last interval, we want to merge them
- Merging: simply set the last value of the last value in the merged list to the max of the two intervals last positions. 
- If not overlapping, simply insert the new interval into the merged list. 

### Time and Space
- Time: O(n log n), due to sorting, and then iterating through the array once
- Space: O(n), the merged list. 

## Non-overlapping Intervals
Given an array of intervals intervals where intervals[i] = [start_i, end_i], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Note: Intervals are non-overlapping even if they have a common point. For example, [1, 3] and [2, 4] are overlapping, but [1, 2] and [2, 3] are non-overlapping.

```python
# Return minimum number to remove to make non-overlapping
# Sort intervals by end time
# init removed, last_end variables
# iterate through remaining intervals, unpack start and end of each interval
# Check if current interval overlaps with the last kept interval, increment removed
# else: move the last_end pointer to end. 
# return removed. 
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        
        intervals.sort(key=lambda x: x[1]) # Sort by end
        removed = 0
        last_end = intervals[0][1]

        for i in range(1, len(intervals)):
            start, end = intervals[i]

            if start < last_end:
                removed += 1
            else: 
                last_end = end
        
        return removed
```

### Key Concepts:
- Sorting by end allow us to handle the edge cases, and allows us to "remove" the correct interval in a greedy way
- Keep track of a last end pointer, and a removed counter
- Loop through remaining intervals, and if the start of current is less than last end, we have an overlap so increment the removed counter
- Else: no overlap, move the last_end pointer to the current end. 

### Time and Space:
- Time: O(n log n), due to sorting
- Space: O(1)


## Meeting Rooms
Given an array of meeting time interval objects consisting of start and end times [[start_1,end_1],[start_2,end_2],...] (start_i < end_i), determine if a person could add all meetings to their schedule without any conflicts.

```python
"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""
# Return True if a person can attend all meetingss in the schedule
# Edgecase, if not intervals
# sort the intervals by start time
# loop starting from the second position
# check if current start is less than previouss end, if so return false
# return True if we make it out of the loop

class Solution:
    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        if not intervals:
            return True
        
        intervals.sort(key=lambda x: x.start)

        for i in range(len(intervals) - 1):
            if intervals[i+1].start < intervals[i].end:
                return False
        
        return True
```

### Key Concepts:
- Sort the intervals by start time
- Check adjacent intervals, if interval + 1 start is less than interval end then return False

### Time and Space:
- Time: O(n log n)
- Space: O(1)

## Meeting Rooms II
Given an array of meeting time interval objects consisting of start and end times [[start_1,end_1],[start_2,end_2],...] (start_i < end_i), find the minimum number of days required to schedule all meetings without any conflicts.

```python
"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""
# Return the number of meeting rooms needed
# initialise two sorte arrays of starting and ending times for the intervals
# init res and count variables, and a s and e pointer for each array
# while s is less than len intervals
# if the start at s is less than end at e, increment the s pointer, and increment the count
# else, a meeting has ended, shift the e pointer and decrement the count
# set res to the max of count and res
# return res

class Solution:
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        start = sorted([i.start for i in intervals])
        end = sorted([i.end for i in intervals])
        max_count, count = 0, 0

        i, j = 0, 0

        while i < len(intervals):
            if start[i] < end[j]:
                i += 1
                count += 1
            else:
                j += 1
                count -= 1
            max_count = max(max_count, count)
        
        return max_count
```

### Key Concepts:
- This can also be solved using a MinHeap
- This solution uses two sorted arrays of start and end times. 
- we loop until we run out of start times
- we check if the current start value is less than the current end value, if so we "start" a meeting and increment the count
- else, if the end value is less than the  current start value, we "end" a meeting and decrement the count and shift the end pointer
- every iteration we update our max_count

### Time and Space:
- Time: O(n log n)
- Space: O(n)