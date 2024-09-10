from itertools import combinations
from collections import defaultdict

class Apriori:
    def __init__(self,transactions:list,min_support=0.2,itemset_limit=5):
        self.min_support = min_support
        self.itemset_limit = itemset_limit 
        self.transactions = transactions
        self.frequent_itemsets = []
        
    def _create_initial_candidates(self):
        """Create one items initial candidates."""
        print(f"Get candidates for length 1.")
        single_items = set(frozenset([item]) for transaction in self.transactions for item in transaction)
        return single_items

        
    def _get_support(self,itemset):
        """Calculate the support of a given itemset."""
        itemset = frozenset(itemset)
        counts = [True  if itemset.issubset(transaction) else False for transaction in self.transactions]
        return sum(counts) / len(transactions)
        
    def _get_frequent_itemsets(self,candidates):
        """Filter candidates to find frequent itemsets."""
        print("Get frequent itemsets.")
        frequent_itemsets = []
        
        candidate_counts = defaultdict(int)
        for candidate in candidates:
            candidate_support = self._get_support(candidate)
            if candidate_support >= self.min_support:
                frequent_itemsets.append((candidate,candidate_support))        
        
        print("Ok!",end='\n\n')
        return frequent_itemsets

    
    def _get_candidates(self,previous_frequent_itemsets, length):
        """Generate candidate itemsets of a specific length."""
        print(f"Get candidates for length {length}")
        candidates = []
        frequent_items = set()
        for itemset, _ in previous_frequent_itemsets:
            frequent_items.update(itemset)
    
        for comb in combinations(frequent_items, length):
            candidates.append(comb)
        
        return candidates

    def _update_frequent_itemsets(self,current_frequent_itemsets):
        if current_frequent_itemsets not in self.frequent_itemsets:
            self.frequent_itemsets.append(current_frequent_itemsets)
            
    def fit(self):
        initial_candidates = self._create_initial_candidates() # single items
        current_frequent_itemsets = self._get_frequent_itemsets(candidates=initial_candidates)
        
        k = 2
        while len(current_frequent_itemsets) > 0:
            print(f"{k} itemset combination is searching...")
            
            if k > self.itemset_limit:
                print("Itemset limit reached.")
                break 
                
            # Add the current frequent itemsets to the list
            self._update_frequent_itemsets(current_frequent_itemsets)
            # Generate next-level candidates from the current frequent itemsets
            candidates = self._get_candidates(current_frequent_itemsets, k)
            
            # If no more candidates can be generated, stop the loop
            if not candidates:
                print("No more candidates can be generated.")
                break
            
            # Filter candidates to get frequent itemsets
            current_frequent_itemsets = self._get_frequent_itemsets(candidates)

            # Increment k to move to the next level of itemsets
            k += 1
            
    def get_frequent_itemsets(self):
        """Return all the frequent itemsets with their support values."""
        return self.frequent_itemsets    
