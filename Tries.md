## Implement Trie (Prefix Tree)

A prefix tree (also known as a trie) is a tree data structure used to efficiently store and retrieve keys in a set of strings. Some applications of this data structure include auto-complete and spell checker systems.

Implement the PrefixTree class:

PrefixTree() Initializes the prefix tree object.
void insert(String word) Inserts the string word into the prefix tree.
boolean search(String word) Returns true if the string word is in the prefix tree (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

```python
# Define a TrieNode class, initialise with a children dictionary and endOfWord boolean as false
# init, just set root to a new trienode()
# insert, set pointer to root, loop through character in word, if char not in current.children, create a trienode. Move the pointer. When out of loop set end of word bool
# search, set pointer, loop thru word, if c not in children, return false, if finish loop, return end of word bool
# starts with, set pointer, loop thru prefix, if c not in children, return false, move pointer, if finish return true
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEnd = False

class PrefixTree:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        # set is end
        curr.isEnd = True

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return curr.isEnd

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return True
```

### Key Concepts:

- Tries are essentially a type of tree that stores words using each character as a node in the tree.
- Nodes are shared between words, example 'app' and 'apple' share the same first 3 nodes.
- When we insert a word into the trie, we need to iterate through each character in the word, and create a node for that character if it does not exist.
- When we reach the end of a word, we want to mark it as a "word" by setting the last character's isEnd/endOfWord value to True. This is so that we can identify words in the search function.
- When we initialise a new TrieNode, we create a children dictionary which will store the key values of it's children nodes. This is so that we can easily check if a particular character exists for a particular node.

### Time and Space Complexity

Time:

- insert: O(n) where n is the length of the word
- search: O(n) where n is the lenth of the word
- startsWith: O(n) where n is the length of the prefix.
  Space:
- O(m) where m is the total number of characters across all words inserted into the trie

## Design Add and Search Word Data Structure

Design a data structure that supports adding new words and searching for existing words.

Implement the WordDictionary class:

void addWord(word) Adds word to the data structure.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.

```python
# Define a TrieNode class, children dict, isend bool
# init root at TrieNode
# addWord: set node as root, loop through word, if character not in node children, create child node using character as key. Set the node to the child node. at the end set isEnd
# Search
# helper dfs function with two parameters, node and index
# Base case: we have processed all characters in the word, if index is equal to len of word, return node is end
# set char to the character at the current index.
# if the ccurrent character is a dot, we need to check all possible children
# for child in children, call dfs, passing in the child and index + 1. If true return True immediately
# if we reach the end of the loop we want to return false
# if we dont have a dot, and the character exists in children, continue searching
# if chatacter in children, return dfs passing in the child node given char and index + 1
# if not, this path doesn't match, return False
# at the end call the dfs function on self.root at index 0.
class TrieNode():
    def __init__(self):
        self.children = {}
        self.isEnd = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.isEnd = True

    def search(self, word: str) -> bool:
        def dfs(node, index):
            # Basecase: if index == len(word) return isEnd
            if index == len(word):
                return node.isEnd

            char = word[index]

            # if char is dot character, recursively search all possible paths
            if char == '.':
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True # return true if we find a match, otherwise keep looking through children
                return False
            else:
                # Search for matching character
                if char in node.children:
                    return dfs(node.children[char], index + 1)
                return False
        return dfs(self.root, 0)
```

### Key Concepts:

- We can use a Trie/Prefix tree to help us solve this problem.
- The addWord function is a standard add function for a trie
- loop through each character in the word, if the character is not in the current nodes children, create it and then shift the pointer.
- When we finish the loop, set the isEnd boolean to true for the node at the current pointer
- The search function utilises recursive DFS to check all possible children if we get a wildcard character.
- Basecase: if the index is equal to the length of the word, we want to return the isEnd boolean which will determine whether we found the word.
- If the current character is a wildcard, we want to loop through each child node and call the dfs helper function, passing in the child and the index + 1.
- If the dfs function returns true, we want to do an Early return which will break the loop
- If the current character is not a wildcard, we simply check if that character is in the nodes children, and call the dfs function on that child with index + 1.

### Time and Space Complexity:

- Time:
- addWord: O(n), where n is the length of the word
- search: O(n), where n is the length of the search word

- Space: O(m), where m is the total number of characters across all input words.

## Word Search 2

Given a 2-D grid of characters board and a list of strings words, return all words that are present in the grid.

For a word to be present it must be possible to form the word with a path in the board with horizontally or vertically neighboring cells. The same cell may not be used more than once in a word.

```python
# Given an m x n board of characters and a list of strings words. return all words on the board.
# add word: set curr pointer, loop through characters, if char not in children, create a TrieNode, move pointer to the child. outside loop set is end
# findWords:
# initialise a root trienode, add each word to the trie
# define: ROWS, COLS, result, and visited variables
# dfs: row, column, node, word
# base cases: row or col less than 0, row or col >= ROWS, COLS, current cell not in trie children, or we have visited before
# add to visited, set node, add element to word
# if node is a word, add it to result
# call dfs in 4 directions
# remove from visited (backtrack)
# outside DFS: nested loop, call dfs for every starting pos. return result.
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isWord = False

    def addWord(self, word):
        curr = self
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.isWord = True

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for w in words:
            root.addWord(w)

        rows, cols = len(board), len(board[0])
        result, visited = set(), set()

        def dfs(r, c, node, letters):
            # Base case:
            if r < 0 or c < 0 or r >= rows or c >= cols or board[r][c] not in node.children or (r, c) in visited:
                return

            # add to visited
            visited.add((r, c))
            node = node.children[board[r][c]]
            letters = letters + board[r][c]
            if node.isWord:
                result.add(letters)

            dfs(r + 1, c, node, letters)
            dfs(r - 1, c, node, letters)
            dfs(r, c + 1, node, letters)
            dfs(r, c - 1, node, letters)
            visited.remove((r, c))

        for r in range(rows):
            for c in range(cols):
                dfs(r, c, root, "")

        return list(result)
```

### Key Concepts:

- We can use a Prefix Tree (Trie) to check if the current path is a valid "prefix" of a word in the input list.
- We define a TrieNode class with an addWord function, then add all words in the input list to the Trie
- We loop through every starting position in the board, and then perform a dfs on that starting position.
- We have multiple basecases:
- Base case 1: Out of bounds, we simply check if the row or column falls outside the board
- Base case 2: current element not in the Trie children. This means that the path we are on is not a prefix or a word in the input list
- Base case 3: we have already visited this element, this can be checked using the visited set. (We could have also set the element in place to a # character)
- Core DFS logic:
- Add current element to visited set
- move the node pointer to the child node in the trie (we already checked that this node should exist)
- we want to add the current element to our letters string
- check if the current node is marked as a word in our trie, if so add it to our result set.
- call DFS in all 4 directions on the board
- remove the current element from visited (backtrack)

# Time and Space Complexity:

- Time: O(m _ n _ 4^L), where m and n are the rows and cols, L is the maximum len of a word in the word list, 4^L represents the maximum number of paths (branching factor of 4 at each step)
- Space: O(N _ L + m _ n)
