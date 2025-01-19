# Arrays & Hashing

## 242. Valid Anagram

### Cheating Method (Not Recommended)
```python
def isAnagram(s: str, t: str) -> bool:
    return sorted(s) == sorted(t)
```

### Better Method
```python
def isAnagram(s: str, t: str) -> bool:
    # If strings are different lengths, they cannot be anagrams.
    if len(s) != len(t):
        return False
    
    countS, countT = {}, {}

    # Count occurrences of each character in both strings
    for i in range(len(s)):
        countS[s[i]] = countS.get(s[i], 0) + 1
        countT[t[i]] = countT.get(t[i], 0) + 1
    
    # If both dictionaries are equal, we have an anagram.
    return countS == countT
```

### Key Concepts

- **Current Character**: 
  - `s[i]` and `t[i]` represent the current characters being processed from strings `s` and `t`.

- **Get Count with Default Value of 0**: 
  - `countS.get(s[i], 0)` retrieves the count of the character `s[i]` from the dictionary `countS`. If the character does not exist, it returns `0`.

#### Example:
- Given a dictionary:
  ```python
  {
      'a': 2
  }
  ```
- Calling `countS.get('a', 0)` returns `2`.
- Calling `countS.get('b', 0)` returns `0`.


## Group Anagrams
Given an array of strings strs, group all anagrams together into sublists. You may return the output in any order.

Example:
```
Input: strs = ["act","pots","tops","cat","stop","hat"]

Output: [["hat"],["act", "cat"],["stop", "pots", "tops"]]
```

### Solution 1. Hash Table
```python
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
  res = defaultdict(list) # Creating a default dictionary called res
  for s in strs:
    count = [0] * 26 # Creates an array of 26 0's to represent the characters in the alphabet. 
    for c in s:
      count[ord(c) - ord('a')] += 1
    res[tuple(count)].append(s)
  return res.values()
```

### Key Concepts

- **Default Dictionary**
  - Using a defaultDict allows you to initialise a new entry with a default value when a key is accessed that does not exist in the dictionary. This means you don't have to check if the key exists before appending to the list. 
  - `dict = defaultdict(list)`: Creates a default dictionary where the type of values are lists. 
  {
    key: [value, value2],
    key2: [value, value2]
  }

- **Counting Occurances**
  `count[ord(c) - ord('a')] += 1`
  - `ord(c)`: this function returns the Unicode point (integrer representation) of the character c. 
  - `ord('a')`: This returns the Unicode code point for the chracter 'a'. 
  - `ord(c) - ord('a')`: This calculates the index for the character c in the count array. For example:
      - If c is 'a', then ord('a') - ord('a') equals 0, so it increments count[0]
      - If c is 'b'. then ord('b') - ord('a') equals 1, so it increments count[1]
      - This continues for all lowercase letters up to 'z', which would increment count[25]
  - `count[ord(c) - ord('a')] += 1`: This increments the count of the character c in the count array. Essentially, it keeps track of how many times each character appears in the string s. 

- **Grouping Anagrams**
  `res[tuple(count)].append(s)`
  - Purpose: This line is grouping the anagrams together based on their character counts.
  - How it works:
    - `tuple(count)`: This converts the count list (which contains the counts of each character) into a tuple. 
    Tuples are hashable and can be used as keys in a dictionary, while lists are not. 
    - `res[tuple(count)]`: This accesses the list in the res dictionary that corresponds to the key tuple(count). If this key does not exist yet, it will create a new entry in the dictionary with an empty list. 
    - `.append(s)`: This appends the string s to the list associated with the key tuple(count). this means that all strings that have the same character count (and thus are anagrams of eachother) will be grouped together in the same list. 

- **Displaying Dictionary Values**
  - `res.values()`: This will simply display the values in the dictionary as a list without the keys. 

  Sure! Let's walk through the example using the words "act" and "cat" step by step, showing how the data is processed, how the counts are calculated, and how the results are added to the `res` dictionary.

### Step-by-Step Example

1. **Initialization**:
   - We start with an empty `res` dictionary: `res = {}`.
   - We will process the strings "act" and "cat".

2. **Processing the word "act"**:
   - Initialize the `count` array: `count = [0] * 26` (which creates an array of 26 zeros).
   - For each character in "act":
     - **Character 'a'**:
       - `ord('a') - ord('a')` = `0`
       - Increment `count[0]`: `count = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
     - **Character 'c'**:
       - `ord('c') - ord('a')` = `2`
       - Increment `count[2]`: `count = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
     - **Character 't'**:
       - `ord('t') - ord('a')` = `19`
       - Increment `count[19]`: `count = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
   - Convert `count` to a tuple: `key = tuple(count) = (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)`.
   - Append "act" to the `res` dictionary:
     - `res[key].append("act")` results in `res = {(1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): ["act"]}`.

3. **Processing the word "cat"**:
   - Reset the `count` array: `count = [0] * 26`.
   - For each character in "cat":
     - **Character 'c'**:
       - `ord('c') - ord('a')` = `2`
       - Increment `count[2]`: `count = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
     - **Character 'a'**:
       - `ord('a') - ord('a')` = `0`
       - Increment `count[0]`: `count = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
     - **Character 't'**:
       - `ord('t') - ord('a')` = `19`
       - Increment `count[19]`: `count = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
   - Convert `count` to a tuple: `key = tuple(count) = (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)`.
   - Append "cat" to the `res` dictionary:
     - `res[key].append("cat")` results in `res = {(1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): ["act", "cat"]}`.

### Final Result
After processing both words, the `res` dictionary will look like this:
```python
{
    (1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): ["act", "cat"]
}
```

This shows that "act" and "cat" are grouped together as they are anagrams of each other.

### My Solution:
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        result = defaultdict(list)

        # iterate through the list of strings
        for i in strs:
            # Initialise a count array for the current string. 
            count = [0] * 26
            # iterate through each character in the string
            for c in i:
                # Determine Characters index to add to the count.
                char_index = ord(c) - ord('a')
                # Increment the value at the characters index in the count list. 
                count[char_index] += 1
            # We need to convert to a Tuple before appending string to count key. [1, 0] -> (1, 0)
            key = tuple(count)
            # Use the tuple (Count of each character in the string) as the key to hold the string group
            result[key].append(i) # Append current string to group using tuple count as key. 
        return result.values()
```


## Top K Elements in List
Given an integer array nums and an integer k, return the k most frequent elements within the array.

The test cases are generated such that the answer is always unique.

You may return the output in any order.

Example 1:

Input: nums = [1,2,2,3,3,3], k = 2

Output: [2,3]
Example 2:

Input: nums = [7,7], k = 1

Output: [7]
Constraints:

1 <= nums.length <= 10^4.
-1000 <= nums[i] <= 1000
1 <= k <= number of distinct elements in nums.

### Solution - Bucket Sort
```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Create HashMap and Frequency List
        countMap = {}
        freqList = [[] for i in range(len(nums) + 1)] # [[], [], []] where len(List) + 1
        # Count Frequencies using HashMap
        for num in nums:
            countMap[num] = 1 + countMap.get(num, 0)
        # Transfer Count to Frequency List
        for num, count in countMap.items():
            freqList[count].append(num) # Append Number to Index of the Count value in countMap where key is num
        # Collect the top K frequent elements (Reverse Iteration on Frequency List)
        res = []
        for i in range(len(freqList) - 1, 0, -1):
            for num in freqList[i]:
                res.append(num)
                if len(res) == k:
                    return res
```

### Key Concepts

- **Frequency List Initialization**
- `freq = [[] for i in range(len(nums) + 1)]`
  - Purpose: This line initializes a list called freq that will hold lists of numbers, where the index of each list corresponds to the frequency of those numbers in the input array nums. 
    - How it works: 
      - `len(nums) + 1`: This creates a list with a size of `len(nums) + 1`. The extra space is to accomodate the case where a number appears `len(nums)' times (the maximum frequency possible). 
      - `[[] for i in range(len(nums) + 1)]`: This is a list comprehension that creates a list of empty lists. 
      Each index in freq will eventually hold the numbers that appear with that specific frequency. 
      - For example, if `nums = [1, 2, 3, 3, 3, 3]`, and the frequencies of the numbers are:
        - 1 appears 1 time
        - 2 appears 2 times
        - 3 appears 3 times
      - The freq list will look like this after processing:
      `freq = [[], [1], [2], [3]]`

- **Reverse Iteration Over Frequency List**
- `for i in range(len(freq) - 1, 0, -1):`
  - Purpose: This line iterates over the freq list in reverse order, starting from the highest frequency down to 1.
    - How it works: 
      - `len(freq) - 1`: This gives the index of the last element in the freq list, which corresponds to the maximum frequency of any number in nums.
      - `0`: This is the stopping condition for the loop, meaning the loop will stop before reaching index 0. 
      - `-1`: This indicates that the loop will decrement i by 1 in each iteration, effectively iterating backwards.
      - The loop will thus check the frequency lists from the highest frequency to the lowest. This is important because we want to collect the most frequent elements first.
      - For example, if freq is `[[], [1], [2], [3]], the loop will iterate over indices 3, 2, and 1, allowing the algorithm to gather the most frequent elements first.

## String Encode and Decode
Design an algorithm to encode a list of strings to a single string. The encoded string is then decoded back to the original list of strings.

Please implement encode and decode

Example 1:

Input: ["neet","code","love","you"]

Output:["neet","code","love","you"]
Example 2:

Input: ["we","say",":","yes"]

Output: ["we","say",":","yes"]
Constraints:

0 <= strs.length < 100
0 <= strs[i].length < 200
strs[i] contains only UTF-8 characters.

### Solution
```python
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string."""
        return ''.join(f"{len(s)}#{s}" for s in strs)

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings."""
        outputList = []
        i = 0
        
        while i < len(s):
            # Find the length of the word (everything before '#')
            j = i
            while s[j] != '#':
                j += 1
            wordLength = int(s[i:j])  # Convert the substring before '#' to an integer
            
            # Extract the word of length `wordLength`
            wordStart = j + 1  # Skip past the '#'
            wordEnd = wordStart + wordLength
            word = s[wordStart:wordEnd]
            
            # Append the word to the output list
            outputList.append(word)
            
            # Move `i` to the next encoded word
            i = wordEnd
        
        return outputList
```

### Key Concepts:

- **Encoding the String**
- `''.join(f"(len(s)#{s})" for s in strs)`, loops through each string in the list and concats the length of the string, with the deliminator "#" and the word itself. 

- **Extract Word Lengthr**
- Note: We do not know how many digits the integer containing the word length will be. For instance, it could be 10, therefore we need to find where the first delimiter is. 
```python
j = i
while s[j] != '#':
    j += 1 # Increment j until we reach the first #
wordLength = int(s[i:j])  # Convert the substring before '#' to an integer. (This substring is the integer denoting the word length)
```

- **Substring Extraction**
- string[startIndex:endIndex]

## Products of Array Discluding Self
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

### My Solution:
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        outputArr = []
        for i in range(len(nums)):
            prefix = nums[:i]
            suffix = nums[i+1:]
            result = math.prod(prefix) * math.prod(suffix)
            outputArr.append(result)
        return outputArr
```

### Key Concepts:

- **Array Slicing**
  - Array slicing in Python allows you to access a subsection or subset of an array (or list) without needing to iterate over each element manually. It’s a concise and efficient way to extract parts of a list.
  - General syntax for slicing is: `array[start:end:step]`
  - start: The index where the slice begins (inclusive). If omitted, it defaults to the beginning of the array (0).
  - end: The index where the slice ends (exclusive). The slice will include elements up to, but not including, this index. if omitted, it defaults to the end of the array. 
  - step: (optional) The interval between elements in the slice. If omitted, it defaults to 1 (every element). You can use a negative step to reverse an array

```python
# Basic Slicing
arr = [1, 2, 3, 4, 5]
sub_array = arr[1:4]  # [2, 3, 4]

# Omitting start or end
sub_array = arr[:3]   # [1, 2, 3]
sub_array = arr[2:]   # [3, 4, 5]

# Using Negative Indices:
sub_array = arr[-3:]   # [3, 4, 5]
sub_array = arr[:-2]   # [1, 2, 3]

# Using Step
sub_array = arr[::2]  # [1, 3, 5]
sub_array = arr[::-1] # [5, 4, 3, 2, 1] (Reversed array)

# Applying slicing to our solution
prefix = nums[:i]   # Extracts elements before index i
suffix = nums[i+1:] # Extracts elements after index i
```

- **Product of an Array**
  - You can use: `math.prod(array)` to get the product of all elements of an array. 

- ### Optimal Solution
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        outputArr = [1] * len(nums)
        
        # Calculate prefix array
        prefix = 1
        for i in range(len(nums)):
            outputArr[i] = prefix
            prefix = prefix * nums[i]

        postfix = 1
        # Calculate postfix array
        for i in range(len(nums) -1, -1, -1):
            outputArr[i] *= postfix
            postfix = postfix * nums[i]

        return outputArr
```

### Key Concepts

- **Prefix and Postfix Products**: The solution uses the concept of prefix and postfix products to calculate the result efficiently.
  ```python
  prefix = 1
  for i in range(len(nums)):
      outputArr[i] = prefix # Insert Prefix
      prefix *= nums[i]  # Update Prefix
  
  postfix = 1
  for i in range(len(nums) - 1, -1, -1):
      outputArr[i] *= postfix # Multiply postfix with prefix value
      postfix *= nums[i] # Update Postfix
  ```

- **In-place Calculation**: The algorithm uses the output array itself to store intermediate results, minimizing space complexity.
  ```python
  outputArr = [1] * len(nums)
  ```

- **Two-pass Approach**: The solution involves two passes through the array - one forward pass for prefix products and one backward pass for postfix products.

- **Time Complexity**: O(n), where n is the length of the input array, as we traverse the array twice.

- **Space Complexity**: O(1) extra space (excluding the output array), as we only use a constant amount of extra space.
  ```python
  # Only constant extra variables used
  prefix = 1
  postfix = 1
  ```

- **Handling Edge Cases**: The algorithm naturally handles cases with zeros in the input array without requiring special treatment.
  ```python
  # No special checks for zeros needed
  outputArr[i] *= postfix
  ```


## Valid Sudoku
You are given a a 9 x 9 Sudoku board board. A Sudoku board is valid if the following rules are followed:

Each row must contain the digits 1-9 without duplicates.
Each column must contain the digits 1-9 without duplicates.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without duplicates.
Return true if the Sudoku board is valid, otherwise return false

Note: A board does not need to be full or be solvable to be valid.

## My initial solution:
```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # Brute force solution - Check each Row, Check Each Column, Check Each Sub-Box
        for row in range(len(board)):
            row_set = set()
            for col in range(len(board[0])):
                if board[row][col] != ".":
                    if board[row][col] in row_set:
                        return False
                    row_set.add(board[row][col])
        
        # Check Each Column
        for col in range(len(board[0])): # len(board[0]) gives number of columns
            col_set = set()
            for row in range(len(board)): #len(board) gives number of rows
                if board[row][col] != ".":
                    if board[row][col] in col_set:
                        return False
                    col_set.add(board[row][col])

        # Check each 3x3 sub-box
        # Create a set to track the numbers in the current 3x3 box.

        # for each 3x3 box, loop through rows and columns within the bounds of that box
        for start_row in range(0,9,3):
            for start_col in range(0,9,3):
                # Define set
                box_set = set()
                # from starting point, loop through each row and column
                for row in range(start_row, start_row + 3):
                    for col in range(start_col, start_col  + 3):
                        if board[row][col] != ".":
                            if board[row][col] in box_set:
                                return False
                            box_set.add(board[row][col])
        
        return True
```

## Optimised Solution
```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # Optimised solution
        rowSet = defaultdict(set)
        colSet = defaultdict(set)
        squareSet = defaultdict(set) # Key is (row // 3, col // 3)

        for row in range(9):
            for col in range(9):
                if board[row][col] != ".":
                    if board[row][col] in rowSet[row] or board[row][col] in colSet[col] or board[row][col] in squareSet[(row // 3, col // 3)]:
                        return False
                    rowSet[row].add(board[row][col])
                    colSet[col].add(board[row][col])
                    squareSet[(row // 3, col // 3)].add(board[row][col])
        return True
```

### Key Concepts

- **Default Dictionary**
  - `defaultdict(set)`: This creates a dictionary where the values are sets. 

- **Square Set**
  - `squareSet[(row // 3, col // 3)]`: This creates a set for each 3x3 sub-box. The key is a tuple representing the starting row and column of the sub-box. 
  - `(row // 3, col // 3)`: This calculates the starting row and column of the sub-box by using integer division. 
    - For example, if we are checking the sub-box starting at row 3 and column 4, `(3 // 3, 4 // 3)` equals `(1, 1)`. This means the sub-box is located in the second row and second column of the overall 9x9 grid. 

- **Set Membership Check**
  - `if board[row][col] in rowSet[row] or board[row][col] in colSet[col] or board[row][col] in squareSet[(row // 3, col // 3)]`: This checks if the current number is already present in the row, column, or sub-box. 
    - If it is, the function returns `False` because the Sudoku rules have been violated. 
    - If it isn't, the function adds the number to the respective set and continues checking the rest of the board. 

## Longest Consecutive Sequence

Given an unsorted array of integers, find the length of the longest consecutive elements sequence. Your algorithm must run in ￼ time.

Example:
Input: nums = [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive sequence is [1, 2, 3, 4]. Therefore its length is 4.

Optimal Solution (HashSet Approach)

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # Step 1: Convert list to a HashSet for O(1) lookups
        nums_set = set(nums)
        longest_subset = 0

        # Step 2: Iterate over the set
        for num in nums_set:
            # Check if num is a potential start of a sequence
            if (num - 1) not in nums_set:
                current_num = num
                subset_count = 1

                # Step 3: Count consecutive elements starting from 'num'
                while (current_num + 1) in nums_set:
                    current_num += 1
                    subset_count += 1

                # Step 4: Update the longest sequence found
                longest_subset = max(longest_subset, subset_count)

        return longest_subset
```

Key Concepts:
	•	HashSet for O(1) lookups:
	•	Convert the list to a hash set using nums_set = set(nums). This allows for quick lookups to see if an element exists in the set, ensuring the solution runs in linear time ￼.
	•	Finding the start of a sequence:
	•	A number is the start of a sequence if (num - 1) is not in the set. This ensures we only start counting from the beginning of a sequence, avoiding redundant work for numbers that are part of an already counted sequence.
	•	Counting the consecutive elements:
	•	After identifying the start of a sequence, increment current_num while (current_num + 1) exists in the set, and track the length of the sequence.
	•	Efficiency:
	•	Each number is processed at most twice (once for checking if it’s the start, and once in the sequence count), ensuring the time complexity is ￼.
