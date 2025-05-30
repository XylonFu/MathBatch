# cores/loader.py
import json
import logging
import os
from typing import List, Dict, Any

from .base import BaseDataLoader

logger = logging.getLogger(__name__)


class DataLoader(BaseDataLoader):
    """Concrete implementation of data loader"""

    def __init__(self, index_field: str, response_field: str):
        self.index_field = index_field
        self.response_field = response_field

    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads input data from file"""
        if not os.path.exists(file_path):
            logger.error(f"Input file not found: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Data loading error: {str(e)}", exc_info=True)
            return []

    def initialize_output_dataset(
            self,
            input_data: List[Dict[str, Any]],
            output_file: str,
            rerun: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Initializes output data structure
        :param input_data: Input dataset
        :param output_file: Output file path
        :param rerun: Whether to rerun all samples
        :return: Initialized output dataset
        """
        # Load existing data if available
        if os.path.exists(output_file) and not rerun:
            try:
                with open(output_file, "r", encoding="utf-8") as file:
                    existing_data = json.load(file)

                # Create index mapping
                index_map = {item[self.index_field]: item for item in existing_data}

                # Merge data preserving existing results
                return [
                    index_map.get(item[self.index_field], item.copy())
                    for item in input_data
                ]
            except Exception:
                logger.warning("Failed to load existing output data, starting from scratch")

        # Fresh run or load failure
        return [item.copy() for item in input_data]
