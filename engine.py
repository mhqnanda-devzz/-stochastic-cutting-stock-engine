# Cutting Stock Engine v0.2 (beta)
#
# Contents:
#   - Generator of MAXIMAL cutting patterns (leftover < smallest item),
#     valid for any ordering of the input items.
#   - Ridge regression (L2) for value function approximation.
#   - Demand transition simulation (multiplicative Gaussian noise, memoryless).
#
# Not yet implemented: a solver for the minimum number of stock pieces
# (LP / column generation).

import numpy as np

# Numerical tolerance for floating-point length comparisons.
_EPS = 1e-9


# Raised for invalid input or when a computational limit is exceeded.
class CuttingStockError(ValueError):
    pass


class StochasticCuttingStockEngine:
    def __init__(self, stock_length: float, lambda_reg: float = 1e-3):
        # Validate the stock length and the regularization strength.
        if not np.isfinite(stock_length) or stock_length <= 0:
            raise CuttingStockError("stock_length harus > 0")
        if lambda_reg < 0:
            raise CuttingStockError("lambda_reg tidak boleh negatif")
        self.stock_length = float(stock_length)
        self.lambda_reg = float(lambda_reg)

    # Generate all maximal cutting patterns, i.e. patterns in which no further
    # item fits into the remaining material.
    # Pattern columns follow the original order of `item_lengths`.
    # Items longer than the stock always have a count of zero.
    def generate_maximal_cutting_patterns(self, item_lengths, max_patterns: int = 100_000) -> list:
        items = [float(x) for x in item_lengths]
        if not items:
            raise CuttingStockError("item_lengths tidak boleh kosong")
        if any((not np.isfinite(x)) or x <= 0 for x in items):
            raise CuttingStockError("semua panjang item harus > 0")

        min_item = min(items)
        # No item fits into the stock, so no pattern exists.
        if min_item > self.stock_length + _EPS:
            return []

        # Sort internally in descending order and keep the original indices so
        # that results can be mapped back. With descending order, the smallest
        # item always has an index >= start_idx, which makes the condition
        # "leftover < smallest item" an exact test for maximality.
        order = sorted(range(len(items)), key=lambda i: -items[i])
        sorted_items = [items[i] for i in order]
        n = len(items)

        found = set()
        current = [0] * n

        def backtrack(remaining, start_idx):
            # Base case: nothing else fits, so the pattern is maximal.
            if remaining < min_item - _EPS:
                if any(current):
                    found.add(tuple(current))
                    # Abort if the number of patterns grows beyond the limit.
                    if len(found) > max_patterns:
                        raise CuttingStockError(
                            f"jumlah pola melebihi max_patterns={max_patterns}")
                return
            # Only consider items from start_idx onward to avoid generating
            # the same combination in different orders.
            for i in range(start_idx, n):
                if remaining >= sorted_items[i] - _EPS:
                    current[i] += 1
                    backtrack(remaining - sorted_items[i], i)
                    current[i] -= 1

        # Recursion depth equals the number of pieces per stock; cap it so the
        # interpreter's recursion limit is never reached.
        if self.stock_length / min_item > 500:
            raise CuttingStockError(
                "stock_length / item terkecil terlalu besar (>500 potongan per stock)")
        backtrack(self.stock_length, 0)

        # Map each pattern back to the original item order.
        result = []
        for pat in sorted(found, reverse=True):
            original = [0] * n
            for pos, orig_idx in enumerate(order):
                original[orig_idx] = pat[pos]
            result.append(original)
        return result

    # Solve the ridge regression problem in closed form:
    #   theta = (A^T A + lambda * I)^(-1) A^T b
    def solve_ridge_value_approximation(self, A_matrix, b_vector) -> np.ndarray:
        A = np.asarray(A_matrix, dtype=float)
        b = np.asarray(b_vector, dtype=float)
        if A.ndim != 2:
            raise CuttingStockError("A_matrix harus 2 dimensi")
        # Treat a 1-D target as a single column vector.
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        if A.shape[0] != b.shape[0]:
            raise CuttingStockError("jumlah baris A dan b harus sama")
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
            raise CuttingStockError("A dan b tidak boleh berisi NaN/inf")

        n = A.shape[1]
        cov = A.T @ A + self.lambda_reg * np.eye(n)
        try:
            # Solving the linear system is preferred over explicit inversion
            # for numerical stability.
            return np.linalg.solve(cov, A.T @ b)
        except np.linalg.LinAlgError as e:
            raise CuttingStockError(
                "matriks singular; naikkan lambda_reg") from e

    # Simulate the next-period demand by applying multiplicative Gaussian noise
    # to each item independently (memoryless). Pass `rng` for reproducible results.
    def simulate_stochastic_transition(self, current_demand, demand_variance: float = 0.1,
                                       rng=None) -> list:
        if demand_variance < 0:
            raise CuttingStockError("demand_variance tidak boleh negatif")
        rng = rng if rng is not None else np.random.default_rng()
        shocks = rng.normal(0.0, demand_variance, size=len(current_demand))
        # Round to integers and enforce a minimum demand of 1 per item.
        return [max(1, int(round(d * (1 + s)))) for d, s in zip(current_demand, shocks)]


if __name__ == "__main__":
    engine = StochasticCuttingStockEngine(stock_length=100.0, lambda_reg=0.005)
    items = [25, 15, 10]
    patterns = engine.generate_maximal_cutting_patterns(items)
    print(f"Pola maksimal: {len(patterns)}  contoh: {patterns[:5]}")

    rng = np.random.default_rng(42)
    A = rng.random((20, 3))          # SAMPLE DATA, not real data
    b = rng.random((20, 1))
    print("Theta (data contoh):", engine.solve_ridge_value_approximation(A, b).ravel())
    print("Demand periode berikut (simulasi):",
          engine.simulate_stochastic_transition(items, rng=rng))
