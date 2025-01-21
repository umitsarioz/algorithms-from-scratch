import heapq

class MaxHeap:
    def __init__(self):
        self.heap = []
    
    # O(log n) - Logarithmic time complexity for heap insert
    def push(self, item: int):
        heapq.heappush(self.heap, -item)  # Negate the item to simulate a max-heap
    
    # O(log n) - Logarithmic time complexity for heap pop
    def pop(self) -> int:
        if not self.heap:
            raise IndexError("pop from empty heap")
        return -heapq.heappop(self.heap)  # Negate again to return the original value
    
    # O(1) - Constant time complexity, as it only accesses the root element
    def peek(self) -> int:
        if not self.heap:
            raise IndexError("peek from empty heap")
        return -self.heap[0]  # Negate to get the original value
    
    # O(1) - Constant time complexity
    def size(self) -> int:
        return len(self.heap)

# Example usage
if __name__ == "__main__":
  max_heap = MaxHeap()
  max_heap.push(10)
  max_heap.push(5)
  max_heap.push(20)
  print(max_heap.pop())  # Output: 20
  print(max_heap.peek())  # Output: 10
