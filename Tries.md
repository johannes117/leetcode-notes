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