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