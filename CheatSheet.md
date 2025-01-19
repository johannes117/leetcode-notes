# Cheat Sheet

## Useful Syntax Tips for Python

### Dictionary Operations

- **Get Count with Default Value**:
  - Use `dict.get(key, default)` to retrieve the value for `key` in a dictionary. If the key does not exist, it returns `default`.
  ```python
  countS.get(s[i], 0)  # Returns the count of s[i], or 0 if not found.
  ```

### Looping Through Strings

- **Accessing Characters by Index**:
  - You can access characters in a string using indexing.
  ```python
  current_char = s[i]  # Access the i-th character of string s.
  ```

### Conditional Statements

- **Checking Lengths**:
  - Use `len()` to check the length of strings before comparing them.
  ```python
  if len(s) != len(t):  # If lengths are different, they cannot be anagrams.
      return False
  ```

### Example Usage

- **Counting Occurrences**:
  - Increment counts in a dictionary for each character in a string.
  ```python
  countS[s[i]] = countS.get(s[i], 0) + 1  # Increment count for character s[i].
  ```

### Return Statements

- **Returning Boolean Values**:
  - You can return the result of a comparison directly.
  ```python
  return countS == countT  # Returns True if both dictionaries are equal.
  ```