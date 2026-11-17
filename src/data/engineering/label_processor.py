"""Ground-Truth Label Processing & Class Imbalance Analysis Module (CP-05).

Provides deterministic ground-truth label encoding, whitespace normalization,
bidirectional integer mappings, and rigorous class imbalance analysis.

STRICT BOUNDARY: Imbalance analysis is diagnostic only. No SMOTE, random oversampling,
or synthetic sample generation is performed in CP-05.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassDistributionItem:
    """Distribution metrics for an individual traffic class."""

    class_name: str
    class_id: int
    count: int
    percentage: float
    is_majority_class: bool
    is_rare_class: bool  # Below minimum threshold for reliable stratification (<10 samples)


@dataclass
class ClassImbalanceReport:
    """Detailed class imbalance diagnostics for an engineered dataset."""

    total_samples: int
    num_classes: int
    class_distributions: List[ClassDistributionItem]
    majority_class_name: str
    majority_class_count: int
    majority_class_percentage: float
    minority_class_name: str
    minority_class_count: int
    minority_class_percentage: float
    imbalance_ratio: float  # majority_count / minority_count
    rare_classes: List[str]
    has_critical_imbalance: bool  # Ratio > 100:1 or presence of rare classes
    warnings: List[str] = field(default_factory=list)

    @property
    def majority_class(self) -> str:
        return self.majority_class_name

    @property
    def majority_count(self) -> int:
        return self.majority_class_count

    @property
    def minority_class(self) -> str:
        return self.minority_class_name

    @property
    def minority_count(self) -> int:
        return self.minority_class_count


@dataclass
class LabelMappingMetadata:
    """Persistent bidirectional mapping between ground-truth labels and integer targets."""

    label_column_name: str
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    classes: List[str]
    total_classes: int


class GroundTruthLabelProcessor:
    """Normalizes, encodes, and analyzes ground-truth intrusion target labels."""

    def __init__(self, rare_class_threshold: int = 10):
        """Initialize label processor.

        Args:
            rare_class_threshold: Minimum observation count below which a class is flagged as rare.
        """
        self.rare_class_threshold = rare_class_threshold
        self.mapping_: Optional[LabelMappingMetadata] = None
        self.is_fitted: bool = False

    @property
    def num_classes(self) -> int:
        """Total number of distinct ground-truth classes."""
        return self.mapping_.total_classes if self.mapping_ else 0

    def fit_transform(
        self,
        label_series: pd.Series,
        column_name: str = "Label",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on valid labels, encode them, and return (encoded_targets, valid_mask)."""
        valid_mask = label_series.notna() & (label_series.astype(str).str.strip() != "") & (label_series.astype(str).str.strip() != "None")
        valid_series = label_series[valid_mask]
        self.fit(valid_series, column_name)
        encoded = self.transform(valid_series)
        return encoded, valid_mask.to_numpy(dtype=bool)

    def compute_imbalance_report(self, label_series: pd.Series) -> ClassImbalanceReport:
        """Alias for analyze_imbalance."""
        return self.analyze_imbalance(label_series)

    def fit(self, label_series: pd.Series, column_name: str = "Label") -> "GroundTruthLabelProcessor":
        """Learn unique class vocabulary and establish deterministic integer encoding.

        Classes are sorted alphabetically to ensure identical mapping across runs.

        Args:
            label_series: Raw ground-truth label Series from training partition.
            column_name: Source column identifier.

        Returns:
            Fitted GroundTruthLabelProcessor instance.
        """
        clean_labels = label_series.dropna().astype(str).str.strip()
        unique_classes = sorted(list(clean_labels.unique()))

        if not unique_classes:
            raise ValueError("Cannot fit label processor: 0 valid class labels found.")

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
        idx_to_class = {idx: cls_name for idx, cls_name in enumerate(unique_classes)}

        self.mapping_ = LabelMappingMetadata(
            label_column_name=column_name,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            classes=unique_classes,
            total_classes=len(unique_classes),
        )
        self.is_fitted = True

        logger.info(
            f"Fitted label processor on {len(clean_labels):,} records. "
            f"Detected {len(unique_classes)} distinct classes: {unique_classes}"
        )
        return self

    def transform(self, label_series: pd.Series) -> np.ndarray:
        """Encode ground-truth label strings to integer targets.

        Unseen classes in validation/test are mapped to -1 and logged with a warning.

        Args:
            label_series: Label Series to encode.

        Returns:
            NumPy 1D array of integer targets.
        """
        if not self.is_fitted or self.mapping_ is None:
            raise RuntimeError("GroundTruthLabelProcessor must be fitted before transform().")

        clean_labels = label_series.astype(str).str.strip().values
        class_to_idx = self.mapping_.class_to_idx

        y_encoded = np.empty(len(clean_labels), dtype=np.int64)
        unseen_classes = set()

        for idx, lbl in enumerate(clean_labels):
            if lbl in class_to_idx:
                y_encoded[idx] = class_to_idx[lbl]
            else:
                y_encoded[idx] = -1
                unseen_classes.add(lbl)

        if unseen_classes:
            logger.warning(
                f"Encountered {len(unseen_classes)} unseen ground-truth labels during transform: {unseen_classes}. "
                "Assigned target ID -1."
            )

        return y_encoded

    def inverse_transform(self, y_encoded: np.ndarray) -> List[str]:
        """Convert integer targets back to original label string representations.

        Args:
            y_encoded: 1D integer target array.

        Returns:
            List of class label strings.
        """
        if not self.is_fitted or self.mapping_ is None:
            raise RuntimeError("GroundTruthLabelProcessor must be fitted before inverse_transform().")

        idx_to_class = self.mapping_.idx_to_class
        return [idx_to_class.get(int(val), "UNKNOWN") for val in y_encoded]

    def analyze_imbalance(self, label_series: pd.Series) -> ClassImbalanceReport:
        """Analyze class frequencies, majority/minority ratios, and rare classes.

        Args:
            label_series: Series of label values to audit.

        Returns:
            ClassImbalanceReport with comprehensive imbalance metrics.
        """
        clean_labels = label_series.dropna().astype(str).str.strip()
        total_count = len(clean_labels)

        if total_count == 0:
            return ClassImbalanceReport(
                total_samples=0,
                num_classes=0,
                class_distributions=[],
                majority_class_name="",
                majority_class_count=0,
                majority_class_percentage=0.0,
                minority_class_name="",
                minority_class_count=0,
                minority_class_percentage=0.0,
                imbalance_ratio=1.0,
                rare_classes=[],
                has_critical_imbalance=False,
                warnings=["Dataset contains 0 valid label observations."],
            )

        counts = Counter(clean_labels)
        sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        majority_name, majority_cnt = sorted_classes[0]
        minority_name, minority_cnt = sorted_classes[-1]

        imbalance_ratio = round(majority_cnt / max(minority_cnt, 1), 2)
        majority_pct = round((majority_cnt / total_count * 100), 2)
        minority_pct = round((minority_cnt / total_count * 100), 2)

        items: List[ClassDistributionItem] = []
        rare_classes: List[str] = []
        warnings: List[str] = []

        for cls_name, cnt in sorted_classes:
            pct = round((cnt / total_count * 100), 2)
            is_maj = cls_name == majority_name
            is_rare = cnt < self.rare_class_threshold

            if is_rare:
                rare_classes.append(cls_name)

            class_id = self.mapping_.class_to_idx.get(cls_name, -1) if self.mapping_ else -1

            items.append(
                ClassDistributionItem(
                    class_name=cls_name,
                    class_id=class_id,
                    count=cnt,
                    percentage=pct,
                    is_majority_class=is_maj,
                    is_rare_class=is_rare,
                )
            )

        if rare_classes:
            warnings.append(
                f"Detected {len(rare_classes)} rare attack classes with < {self.rare_class_threshold} samples: "
                f"{rare_classes}. Stratified splitting may require minimum grouping adjustments."
            )

        if imbalance_ratio > 50.0:
            warnings.append(
                f"Severe class imbalance detected: Majority class '{majority_name}' outweighs "
                f"minority class '{minority_name}' by a ratio of {imbalance_ratio}:1. "
                "Models will require balanced class weighting or threshold tuning in CP-06."
            )

        return ClassImbalanceReport(
            total_samples=total_count,
            num_classes=len(sorted_classes),
            class_distributions=items,
            majority_class_name=majority_name,
            majority_class_count=majority_cnt,
            majority_class_percentage=majority_pct,
            minority_class_name=minority_name,
            minority_class_count=minority_cnt,
            minority_class_percentage=minority_pct,
            imbalance_ratio=imbalance_ratio,
            rare_classes=rare_classes,
            has_critical_imbalance=(imbalance_ratio > 100.0 or len(rare_classes) > 0),
            warnings=warnings,
        )
