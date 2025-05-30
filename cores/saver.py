# cores/saver.py
import json
import logging
import os
from typing import List, Dict, Any

from .base import BaseDataSaver

logger = logging.getLogger(__name__)


class DataSaver(BaseDataSaver):
    """Concrete implementation of data saver"""

    def __init__(self, output_file: str):
        self.output_file = output_file
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    def save_data(self, data: List[Dict[str, Any]]) -> bool:
        """Saves data to file"""
        try:
            with open(self.output_file, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Data saving failed: {str(e)}", exc_info=True)
            return False

    def save_checkpoint(self, data: List[Dict[str, Any]]) -> bool:
        """Saves temporary checkpoint file"""
        temp_file = f"{self.output_file}.tmp"
        return self.save_data(data)
