from itertools import combinations

class Apriori:
    def __init__(self, min_support, min_confidence, itemset_limit=None):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemset_limit = itemset_limit
        self.frequent_itemsets = {}  # Changed from set() to dict() to store support values
        self.rules = []

    def _find_unique_items_in_all_transactions(self):
        unique_items = set(item for transaction in self.transactions_list for item in transaction)
        return unique_items

    def _find_candidates(self, elem_count):
        # Generate itemset combinations of size `elem_count`
        candidates = set(frozenset(itemset) for itemset in combinations(self.unique_items, elem_count))
        return candidates

    def _prepare_transactions(self, transactions):
        self.transactions = transactions
        self.transactions_list = [set(t) for t in transactions]  # Convert each transaction to a set
        self.unique_items = self._find_unique_items_in_all_transactions()
        # Set itemset limit to the number of unique items if not provided
        self.itemset_limit = len(self.unique_items) if self.itemset_limit is None else self.itemset_limit

    def _calculate_support(self, itemset):
        transaction_count = len(self.transactions_list)
        subset_count = sum(1 for transaction in self.transactions_list if itemset.issubset(transaction))
        support = subset_count / transaction_count
        return support

    def _filter_frequent_itemsets(self, candidates):
        frequent_itemsets = {}
        for candidate in candidates:
            support = self._calculate_support(candidate)
            if support >= self.min_support:
                frequent_itemsets[candidate] = support
        return frequent_itemsets

    def _find_frequent_itemsets(self):
        for k in range(1, self.itemset_limit + 1):
            print("Checking itemsets of length:", k)
            candidates = self._find_candidates(k)
            curr_itemsets = self._filter_frequent_itemsets(candidates)
            
            if not curr_itemsets:
                print("No frequent itemsets found for size", k)
                break  # Exit if no frequent itemsets are found
            
            print("Current frequent itemsets:", curr_itemsets)
            self.frequent_itemsets.update(curr_itemsets)

    def _generate_association_rules(self):
        """Generate all possible association rules from the frequent itemsets."""
        for itemset, itemset_support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue  # No rules can be generated from single-item sets
            
            # Generate all non-empty subsets of the itemset (possible antecedents)
            for antecedent_size in range(1, len(itemset)):
                for antecedent in combinations(itemset, antecedent_size):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if consequent:
                        antecedent_support = self.frequent_itemsets.get(antecedent, 0)
                        if antecedent_support > 0:  # To avoid division by zero
                            confidence = itemset_support / antecedent_support
                            if confidence >= self.min_confidence:
                                rule = {
                                    'antecedent': antecedent,
                                    'consequent': consequent,
                                    'confidence': confidence,
                                    'support': itemset_support
                                }
                                self.rules.append(rule)

    def fit(self, transactions):
        self._prepare_transactions(transactions)
        self._find_frequent_itemsets()
        self._generate_association_rules()

    def get_frequent_itemsets(self):
        if not self.frequent_itemsets:
            print("Frequent itemsets not found. Please fit the model first.")
        return self.frequent_itemsets

    def get_rules(self):
        if not self.rules:
            print("No rules generated. Please fit the model and generate association rules.")
        return self.rules

if __name__ == "__main__":
    apr = Apriori(min_support=0.3, min_confidence=0.7)
    
    transactions_by_name = [
    ['The Matrix', 'Inception', 'Interstellar'],
    ['Inception', 'Interstellar'],
    ['The Matrix', 'Interstellar'],
    ['The Matrix', 'Inception'],
    ['Inception', 'Interstellar']
    ]
    
    apr.fit(transactions_by_genre)
    frequent_itemsets = apr.get_frequent_itemsets()
    rules = apr.get_rules()
    
    print("Frequent Itemsets:", frequent_itemsets)
    print("Association Rules:")
    for rule in rules:
        print(f"Rule: {rule['antecedent']} -> {rule['consequent']}, Confidence: {rule['confidence']:.2f}, Support: {rule['support']:.2f}")
