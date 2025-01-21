from typing import List, Tuple, Dict
import math
import heapq

class DynamicProgrammingAlgorithms:
    """Dynamic Programming Algorithms for Knapsack, Traveling Salesman, and Dijkstra."""

    @staticmethod
    def knapsack(capacity: int, weights: List[int], values: List[int], n: int) -> int:
        """
        Knapsack Problem (0/1 Knapsack Problem).
        Time Complexity: O(N * W) where N is the number of items and W is the capacity of the knapsack.
        Space Complexity: O(N * W)

        :param capacity: Maximum capacity of the knapsack.
        :param weights: List of item weights.
        :param values: List of item values.
        :param n: Number of items.
        :return: Maximum value that can be obtained with the given knapsack capacity.
        """
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        return dp[n][capacity]

    @staticmethod
    def traveling_salesman(n: int, dist: List[List[int]]) -> int:
        """
        Traveling Salesman Problem (TSP) using Dynamic Programming with Bit Masking.
        Time Complexity: O(N^2 * 2^N)
        Space Complexity: O(N * 2^N)

        :param n: Number of cities.
        :param dist: 2D list of distances between cities.
        :return: Minimum cost to visit all cities and return to the starting city.
        """
        dp = [[math.inf] * (1 << n) for _ in range(n)]  # dp[i][mask] means the minimum cost to reach city `i` with visited cities in `mask`
        dp[0][1] = 0  # Start at city 0 with only city 0 visited (mask = 1)

        for mask in range(1, 1 << n):
            for u in range(n):
                if (mask & (1 << u)) == 0:
                    continue
                for v in range(n):
                    if (mask & (1 << v)) == 0:  # If `v` has not been visited
                        dp[v][mask | (1 << v)] = min(dp[v][mask | (1 << v)], dp[u][mask] + dist[u][v])

        return min(dp[i][(1 << n) - 1] + dist[i][0] for i in range(1, n))

    @staticmethod
    def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
        """
        Dijkstra's Shortest Path Algorithm.
        Time Complexity: O(E log V), where E is the number of edges and V is the number of vertices.
        Space Complexity: O(V)

        :param graph: Adjacency list representation of the graph.
        :param start: Starting vertex.
        :return: Dictionary with the shortest distance from the start vertex to each other vertex.
        """
        dist = {node: math.inf for node in graph}  # Initialize all distances as infinity
        dist[start] = 0
        pq = [(0, start)]  # (distance, node)

        while pq:
            current_dist, u = heapq.heappop(pq)  # Pop the node with the smallest distance
            if current_dist > dist[u]:
                continue
            for v, weight in graph[u]:
                alt = current_dist + weight
                if alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(pq, (alt, v))

        return dist

    @staticmethod
    def longest_increasing_subsequence(nums: List[int]) -> int:
        """
        Longest Increasing Subsequence (LIS).
        Time Complexity: O(N^2)
        Space Complexity: O(N)

        :param nums: List of integers.
        :return: Length of the longest increasing subsequence.
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # dp[i] will store the length of LIS ending at index i

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)

    @staticmethod
    def coin_change(coins: List[int], amount: int) -> int:
        """
        Coin Change Problem.
        Time Complexity: O(N * A), where N is the number of coins and A is the amount.
        Space Complexity: O(A)

        :param coins: List of coin denominations.
        :param amount: Target amount.
        :return: The minimum number of coins needed to make up the given amount.
                 If no combination is possible, return -1.
        """
        dp = [math.inf] * (amount + 1)
        dp[0] = 0  # Base case: 0 coins are needed to make amount 0

        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] != math.inf else -1


# Example Usage:
if __name__ == "__main__":
    # Knapsack Problem Example
    knapsack_capacity = 50
    weights = [10, 20, 30]
    values = [60, 100, 120]
    n = len(weights)
    print("Knapsack Maximum Value:", DynamicProgrammingAlgorithms.knapsack(knapsack_capacity, weights, values, n))

    # Traveling Salesman Problem Example
    dist = [
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 5],
        [20, 25, 30, 0, 15],
        [25, 30, 5, 15, 0]
    ]
    n = len(dist)
    print("TSP Minimum Cost:", DynamicProgrammingAlgorithms.traveling_salesman(n, dist))

    # Dijkstra Example
    graph = {
        0: [(1, 4), (2, 1)],
        1: [(2, 2), (3, 5)],
        2: [(3, 1)],
        3: []
    }
    start = 0
    print("Dijkstra Shortest Path:", DynamicProgrammingAlgorithms.dijkstra(graph, start))

    # Longest Increasing Subsequence Example
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print("Longest Increasing Subsequence Length:", DynamicProgrammingAlgorithms.longest_increasing_subsequence(nums))

    # Coin Change Example
    coins = [1, 2, 5]
    amount = 11
    print("Coin Change Minimum Coins:", DynamicProgrammingAlgorithms.coin_change(coins, amount))
