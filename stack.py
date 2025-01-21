class Stack:
    # Last in First Out (LIFO) 
    def __init__(self):
        self.items = []
    
    # O(1) - Constant time complexity
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    # O(1) - Constant time complexity
    def push(self, item: int):
        self.items.append(item)
    
    # O(1) - Constant time complexity
    def pop(self) -> int:
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()
    
    # O(1) - Constant time complexity
    def peek(self) -> int:
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]
    
    # O(1) - Constant time complexity
    def size(self) -> int:
        return len(self.items)

# Example usage
if __name__ == "__main__":
  stack = Stack()
  stack.push(10)
  stack.push(20)
  print(stack.pop())  # Output: 20
  print(stack.peek())  # Output: 10
