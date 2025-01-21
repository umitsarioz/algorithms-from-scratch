import heapq

class MinHeap:
    def __init__(self):
        self.heap = []
    
    # O(log n) - Logarithmic time complexity for heap insert
    def push(self, item: int):
        heapq.heappush(self.heap, item)
    
    # O(log n) - Logarithmic time complexity for heap pop
    def pop(self) -> int:
        if not self.heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self.heap)
    
    # O(1) - Constant time complexity, as it only accesses the root element
    def peek(self) -> int:
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0]
    
    # O(1) - Constant time complexity
    def size(self) -> int:
        return len(self.heap)

# Example usage
if __name__ == "__main__":
  min_heap = MinHeap()
  min_heap.push(10)
  min_heap.push(5)
  min_heap.push(20)
  print(min_heap.pop())  # Output: 5
  print(min_heap.peek())  # Output: 10
