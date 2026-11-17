
"""Register and access dataset-specific adapters."""

from src.dataset_preparation.adapters.kth_adapter import KTHAdapter
from src.dataset_preparation.adapters.ucf101_adapter import UCF101Adapter


class DatasetRegistry:
    """Central registry for supported dataset adapters."""

    def __init__(self):
        self._adapters = {}

    def register(self, name: str, adapter) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Dataset name must be non-empty.")

        if not callable(getattr(adapter, "scan", None)):
            raise TypeError("Adapter must provide a scan() method.")

        key = name.strip().lower()

        if key in self._adapters:
            raise ValueError(f"Dataset already registered: {name}")

        self._adapters[key] = adapter

    def scan(self, name: str):
        key = name.strip().lower()

        if key not in self._adapters:
            raise KeyError(f"Dataset is not registered: {name}")

        return self._adapters[key].scan()

    def available_datasets(self) -> list[str]:
        return sorted(self._adapters)


def create_default_registry(
    kth_root,
    ucf101_root,
    ucf101_split: int = 1,
) -> DatasetRegistry:
    """Create a registry containing the KTH and UCF101 adapters."""

    registry = DatasetRegistry()

    registry.register("KTH", KTHAdapter(kth_root))
    registry.register(
        "UCF101",
        UCF101Adapter(ucf101_root, split_number=ucf101_split),
    )

    return registry
