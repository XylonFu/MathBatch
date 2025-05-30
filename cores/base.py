# cores/base.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders"""

    @abstractmethod
    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads input data from file"""
        pass

    @abstractmethod
    def initialize_output_dataset(
            self,
            input_data: List[Dict[str, Any]],
            output_file: str,
            rerun: bool = False
    ) -> List[Dict[str, Any]]:
        """Initializes output data structure"""
        pass


class BaseDataFilter(ABC):
    """Abstract base class for data filters"""

    @abstractmethod
    def filter_unprocessed_items(
            self,
            data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filters items that need processing"""
        pass


class BaseDataProcessor(ABC):
    """Abstract base class for data processors"""

    def __init__(self, field_name: str):
        self.field_name = field_name

    @abstractmethod
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single data item"""
        pass

    @abstractmethod
    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List]:
        """Processes a batch of data items"""
        pass


class BaseDataBatcher(ABC):
    """Abstract base class for data batchers"""

    @abstractmethod
    def process_batches(
            self,
            data: List[Dict[str, Any]],
            response_field: str
    ) -> List[Dict[str, Any]]:
        """Processes data and generates responses"""
        pass


class BaseDataSaver(ABC):
    """Abstract base class for data savers"""

    @abstractmethod
    def save_data(self, data: List[Dict[str, Any]]) -> bool:
        """Saves data to file"""
        pass

    @abstractmethod
    def save_checkpoint(self, data: List[Dict[str, Any]]) -> bool:
        """Saves temporary checkpoint file"""
        pass
