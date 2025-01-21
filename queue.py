class Queue:
    # First in First Out (FIFO) 
    def __init__(self):
        self.items = []
    
    # O(1) - Constant time complexity
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    # O(1) - Constant time complexity
    def enqueue(self, item: int):
        self.items.append(item)
    
    # O(n) - Linear time complexity, as it needs to shift the elements
    def dequeue(self) -> int:
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.items.pop(0)
    
    # O(1) - Constant time complexity
    def front(self) -> int:
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self.items[0]
    
    # O(1) - Constant time complexity
    def size(self) -> int:
        return len(self.items)

# Example usage
if __name__ == "__main__":
  queue = Queue()
  queue.enqueue(10)
  queue.enqueue(20)
  print(queue.dequeue())  # Output: 10
  print(queue.front())  # Output: 20
