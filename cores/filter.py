# cores/filter.py
import logging
from typing import List, Dict, Any

from .base import BaseDataFilter

logger = logging.getLogger(__name__)


class DataFilter(BaseDataFilter):
    """Concrete implementation of data filter"""

    def __init__(self, response_field: str, rerun: bool = False):
        self.response_field = response_field
        self.rerun = rerun

    def filter_unprocessed_items(
            self,
            data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filters items that need processing"""
        if self.rerun:
            return data

        return [
            item for item in data
            if self.response_field not in item or not item[self.response_field]
        ]
