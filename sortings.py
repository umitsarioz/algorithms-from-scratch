from typing import List


class SortingAlgorithms:
    """Class to implement various sorting algorithms"""

    @staticmethod
    def bubble_sort(arr: List[int]) -> List[int]:
        """
        Bubble Sort Algorithm.
        Best Time Complexity: O(N) - when the array is already sorted.
        Worst Time Complexity: O(N^2) - when the array is sorted in reverse order.
        Space Complexity: O(1)

        :param arr: List of integers to be sorted.
        :return: Sorted list.
        """
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]  # Swap elements
                    swapped = True
            if not swapped:
                break  # Exit early if no elements were swapped
        return arr

    @staticmethod
    def merge_sort(arr: List[int]) -> List[int]:
        """
        Merge Sort Algorithm.
        Time Complexity: O(N log N)
        Space Complexity: O(N)

        :param arr: List of integers to be sorted.
        :return: Sorted list.
        """
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        return SortingAlgorithms.merge(SortingAlgorithms.merge_sort(left), SortingAlgorithms.merge_sort(right))

    @staticmethod
    def merge(left: List[int], right: List[int]) -> List[int]:
        """Merge two sorted lists."""
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    @staticmethod
    def quick_sort(arr: List[int]) -> List[int]:
        """
        Quick Sort Algorithm.
        Best Time Complexity: O(N log N)
        Worst Time Complexity: O(N^2) - with bad pivot selection.
        Average Time Complexity: O(N log N)
        Space Complexity: O(log N)

        :param arr: List of integers to be sorted.
        :return: Sorted list.
        """
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return SortingAlgorithms.quick_sort(left) + middle + SortingAlgorithms.quick_sort(right)

    @staticmethod
    def insertion_sort(arr: List[int]) -> List[int]:
        """
        Insertion Sort Algorithm.
        Best Time Complexity: O(N) - when the array is already sorted.
        Worst Time Complexity: O(N^2) - when the array is sorted in reverse order.
        Space Complexity: O(1)

        :param arr: List of integers to be sorted.
        :return: Sorted list.
        """
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    @staticmethod
    def selection_sort(arr: List[int]) -> List[int]:
        """
        Selection Sort Algorithm.
        Time Complexity: O(N^2) for all cases.
        Space Complexity: O(1)

        :param arr: List of integers to be sorted.
        :return: Sorted list.
        """
        for i in range(len(arr)):
            min_idx = i
            for j in range(i + 1, len(arr)):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr

    @staticmethod
    def heap_sort(arr: List[int]) -> List[int]:
        """
        Heap Sort Algorithm.
        Time Complexity: O(N log N) for all cases.
        Space Complexity: O(1)

        :param arr: List of integers to be sorted.
        :return: Sorted list.
        """
        def heapify(arr: List[int], n: int, i: int):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and arr[left] > arr[largest]:
                largest = left
            if right < n and arr[right] > arr[largest]:
                largest = right
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]  # Swap
                heapify(arr, n, largest)

        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]  # Swap root with the last element
            heapify(arr, i, 0)
        return arr

    @staticmethod
    def counting_sort(arr: List[int], max_val: int) -> List[int]:
        """
        Counting Sort Algorithm.
        Time Complexity: O(N + K), where N is the size of the array and K is the range of input.
        Space Complexity: O(N + K)

        :param arr: List of integers to be sorted.
        :param max_val: The maximum value in the input array.
        :return: Sorted list.
        """
        count = [0] * (max_val + 1)
        for num in arr:
            count[num] += 1
        sorted_arr = []
        for i in range(len(count)):
            sorted_arr.extend([i] * count[i])
        return sorted_arr


# Example Usage
if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    print("Bubble Sort:", SortingAlgorithms.bubble_sort(arr.copy()))
    print("Merge Sort:", SortingAlgorithms.merge_sort(arr.copy()))
    print("Quick Sort:", SortingAlgorithms.quick_sort(arr.copy()))
    print("Insertion Sort:", SortingAlgorithms.insertion_sort(arr.copy()))
    print("Selection Sort:", SortingAlgorithms.selection_sort(arr.copy()))
    print("Heap Sort:", SortingAlgorithms.heap_sort(arr.copy()))
    print("Counting Sort:", SortingAlgorithms.counting_sort(arr.copy(), max(arr)))
