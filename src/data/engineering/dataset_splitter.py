"""Dataset Splitting & Data Leakage Prevention Module (CP-05).

Executes reproducible partitioning into Train, Validation, and Test subsets BEFORE
any preprocessor fitting:
  - Default proportions: 70% Train, 15% Validation, 15% Test (Seed: 42)
  - Supports Stratified splitting to preserve attack class ratios
  - Supports Group-Isolated splitting to prevent identical feature records from
    crossing partition boundaries
  - Guarantees complete record accounting: Train + Val + Test = Total Records
"""

from collections import Counter
from dataclasses import dataclass, field
import enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SplittingStrategy(str, enum.Enum):
    """Supported partitioning strategies."""

    STRATIFIED = "stratified"
    GROUP_ISOLATED = "group_isolated"
    RANDOM = "random"


@dataclass
class PartitionIndices:
    """Zero-based integer row indices for each supervised partition."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    total_samples: int


@dataclass
class SplittingAuditReport:
    """Auditable metrics and class distributions across all generated partitions."""

    strategy: str
    random_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    train_percentage: float
    val_percentage: float
    test_percentage: float
    class_distribution_train: Dict[str, int]
    class_distribution_val: Dict[str, int]
    class_distribution_test: Dict[str, int]
    missing_classes_in_val: List[str]
    missing_classes_in_test: List[str]
    group_leakage_detected: bool
    notes: List[str] = field(default_factory=list)


class ReproducibleDatasetSplitter:
    """Partitions supervised datasets reproducibly while enforcing anti-leakage invariants."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        strategy: SplittingStrategy = SplittingStrategy.STRATIFIED,
    ):
        """Initialize splitting ratios and seed."""
        total = round(train_ratio + val_ratio + test_ratio, 4)
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(
                f"Split ratios must sum to 1.0. Observed: {train_ratio} + {val_ratio} + {test_ratio} = {total}"
            )

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.strategy = strategy

    def split(
        self,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
        strategy: Optional[SplittingStrategy] = None,
        df: Optional[pd.DataFrame] = None,
        target_series: Optional[pd.Series] = None,
    ) -> Tuple[PartitionIndices, SplittingAuditReport]:
        """Generate reproducible partition indices and audit reports.

        Args:
            y: Optional 1D array of ground-truth labels (or class IDs).
            groups: Optional list of group fingerprints for group-isolated splitting.
            strategy: Optional SplittingStrategy to apply.
            df: Optional DataFrame if y is omitted.
            target_series: Optional target Series if y is omitted.

        Returns:
            Tuple of (PartitionIndices, SplittingAuditReport).
        """
        strat = strategy or self.strategy

        if y is None:
            if target_series is not None:
                y_arr = target_series.to_numpy()
            elif df is not None:
                label_col = "Label" if "Label" in df.columns else df.columns[-1]
                y_arr = df[label_col].to_numpy()
            else:
                raise ValueError("Must provide 'y', 'target_series', or 'df' to split.")
        elif isinstance(y, (pd.Series, list)):
            y_arr = np.asarray(y)
        else:
            y_arr = y

        if isinstance(groups, np.ndarray):
            groups = [str(g) for g in groups]

        n_samples = len(y_arr)
        if n_samples < 3:
            raise ValueError(f"Insufficient samples to partition into 3 splits: {n_samples} provided.")

        rng = np.random.RandomState(self.random_seed)
        notes: List[str] = []

        # Check class counts for stratification feasibility
        counts = Counter(y_arr)
        min_class_count = min(counts.values()) if counts else 0
        can_stratify = min_class_count >= 3  # Needs at least 1 sample per partition (train, val, test)

        actual_strategy = strat
        if strat == SplittingStrategy.STRATIFIED and not can_stratify:
            logger.warning(
                f"Minimum class count ({min_class_count}) is < 3. "
                "Cannot perform pure 3-way stratification. Falling back to random splitting."
            )
            notes.append(
                f"Stratification adjusted: rare class with {min_class_count} sample(s) cannot span 3 partitions. "
                "Performed random partitioning to preserve rare observations in training."
            )
            actual_strategy = SplittingStrategy.RANDOM

        if strat == SplittingStrategy.GROUP_ISOLATED and groups is not None:
            # Group-isolated splitting: Group all identical feature records together
            train_idx, val_idx, test_idx = self._group_isolated_split(groups, rng)
            actual_strategy = SplittingStrategy.GROUP_ISOLATED
            notes.append(
                "Group-isolated splitting applied: records sharing identical feature fingerprints "
                "are strictly confined to a single partition."
            )
        elif actual_strategy == SplittingStrategy.STRATIFIED:
            train_idx, val_idx, test_idx = self._stratified_split(y_arr, rng)
        else:
            train_idx, val_idx, test_idx = self._random_split(n_samples, rng)

        # Invariant checks: No overlaps and complete accounting
        set_train, set_val, set_test = set(train_idx), set(val_idx), set(test_idx)
        assert len(set_train.intersection(set_val)) == 0, "Data leakage: Train and Validation indices overlap!"
        assert len(set_train.intersection(set_test)) == 0, "Data leakage: Train and Test indices overlap!"
        assert len(set_val.intersection(set_test)) == 0, "Data leakage: Validation and Test indices overlap!"
        assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples, "Row accounting loss in splitting!"

        # Check group leakage if groups were provided
        group_leakage = False
        if groups is not None:
            train_groups = {groups[i] for i in train_idx}
            val_groups = {groups[i] for i in val_idx}
            test_groups = {groups[i] for i in test_idx}
            if train_groups.intersection(val_groups) or train_groups.intersection(test_groups) or val_groups.intersection(test_groups):
                group_leakage = True
                notes.append("Warning: Identical feature groups are present across multiple partitions.")

        # Class distributions
        dist_train = {str(k): int(v) for k, v in Counter(y_arr[train_idx]).items()}
        dist_val = {str(k): int(v) for k, v in Counter(y_arr[val_idx]).items()}
        dist_test = {str(k): int(v) for k, v in Counter(y_arr[test_idx]).items()}

        all_classes = set(counts.keys())
        missing_val = [str(c) for c in all_classes if c not in Counter(y_arr[val_idx])]
        missing_test = [str(c) for c in all_classes if c not in Counter(y_arr[test_idx])]

        if missing_val:
            notes.append(f"Classes missing from validation partition due to low frequency: {missing_val}")
        if missing_test:
            notes.append(f"Classes missing from test partition due to low frequency: {missing_test}")

        indices = PartitionIndices(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            total_samples=n_samples,
        )

        report = SplittingAuditReport(
            strategy=actual_strategy.value,
            random_seed=self.random_seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            total_samples=n_samples,
            train_count=len(train_idx),
            val_count=len(val_idx),
            test_count=len(test_idx),
            train_percentage=round(len(train_idx) / n_samples * 100, 2),
            val_percentage=round(len(val_idx) / n_samples * 100, 2),
            test_percentage=round(len(test_idx) / n_samples * 100, 2),
            class_distribution_train=dist_train,
            class_distribution_val=dist_val,
            class_distribution_test=dist_test,
            missing_classes_in_val=missing_val,
            missing_classes_in_test=missing_test,
            group_leakage_detected=group_leakage,
            notes=notes,
        )

        return indices, report

    def _stratified_split(self, y: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform two-stage stratified split."""
        n_samples = len(y)
        all_indices = np.arange(n_samples)

        # Stage 1: Split into Train vs Temp (Val + Test)
        temp_ratio = self.val_ratio + self.test_ratio
        train_idx, temp_idx = train_test_split(
            all_indices,
            test_size=temp_ratio,
            random_state=rng.randint(0, 100000),
            stratify=y,
        )

        # Stage 2: Split Temp into Val and Test
        temp_y = y[temp_idx]
        counts_temp = Counter(temp_y)
        can_stratify_temp = min(counts_temp.values()) >= 2

        val_fraction_of_temp = self.val_ratio / temp_ratio

        if can_stratify_temp:
            val_sub_idx, test_sub_idx = train_test_split(
                np.arange(len(temp_idx)),
                test_size=(1.0 - val_fraction_of_temp),
                random_state=rng.randint(0, 100000),
                stratify=temp_y,
            )
        else:
            val_sub_idx, test_sub_idx = train_test_split(
                np.arange(len(temp_idx)),
                test_size=(1.0 - val_fraction_of_temp),
                random_state=rng.randint(0, 100000),
            )

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    def _random_split(self, n_samples: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform two-stage random split."""
        all_indices = np.arange(n_samples)
        temp_ratio = self.val_ratio + self.test_ratio

        train_idx, temp_idx = train_test_split(
            all_indices,
            test_size=temp_ratio,
            random_state=rng.randint(0, 100000),
        )

        val_fraction_of_temp = self.val_ratio / temp_ratio
        val_sub_idx, test_sub_idx = train_test_split(
            np.arange(len(temp_idx)),
            test_size=(1.0 - val_fraction_of_temp),
            random_state=rng.randint(0, 100000),
        )

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    def _group_isolated_split(
        self,
        groups: List[str],
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Partition records so that identical group fingerprints never cross partitions."""
        from collections import defaultdict
        group_to_indices = defaultdict(list)
        for idx, g in enumerate(groups):
            group_to_indices[g].append(idx)

        unique_groups = list(group_to_indices.keys())
        rng.shuffle(unique_groups)

        total_samples = len(groups)
        target_train = int(total_samples * self.train_ratio)
        target_val = int(total_samples * self.val_ratio)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        cur_train = 0
        cur_val = 0

        for g in unique_groups:
            indices = group_to_indices[g]
            sz = len(indices)
            if cur_train + sz <= target_train or (cur_train == 0):
                train_indices.extend(indices)
                cur_train += sz
            elif cur_val + sz <= target_val or (cur_val == 0):
                val_indices.extend(indices)
                cur_val += sz
            else:
                test_indices.extend(indices)

        # Fallback if any partition is empty
        if not val_indices and len(test_indices) > 1:
            val_indices.append(test_indices.pop())
        if not test_indices and len(val_indices) > 1:
            test_indices.append(val_indices.pop())

        return np.sort(np.array(train_indices, dtype=np.int64)), np.sort(np.array(val_indices, dtype=np.int64)), np.sort(np.array(test_indices, dtype=np.int64))
