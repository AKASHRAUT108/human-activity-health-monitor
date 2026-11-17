
"""Dataset-specific adapters."""

from src.dataset_preparation.adapters.kth_adapter import KTHAdapter
from src.dataset_preparation.adapters.ucf101_adapter import UCF101Adapter

__all__ = ["KTHAdapter", "UCF101Adapter"]
