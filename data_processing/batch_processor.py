# data_processing/batch_processor.py
import logging
from typing import List, Dict, Any

from models.base_model import BaseInferenceModel
from .base_processor import BaseBatchProcessor, BaseDataProcessor

logger = logging.getLogger(__name__)


class BatchProcessor(BaseBatchProcessor):
    """Concrete implementation of batch processor"""

    def __init__(self, model: BaseInferenceModel, batch_size: int = 500):
        self.model = model
        self.batch_size = batch_size

    def process_batches(
            self,
            data: List[Dict[str, Any]],
            data_processor: BaseDataProcessor,
            response_field: str
    ) -> List[Dict[str, Any]]:
        """Processes data and generates responses"""
        # Process in batches
        for start_index in range(0, len(data), self.batch_size):
            batch = data[start_index:start_index + self.batch_size]

            # Prepare model inputs
            model_inputs = data_processor.process_batch(batch)

            # Get model responses
            responses = self.model.generate_responses(
                model_inputs["prompts"],
                model_inputs["images"]
            )

            # Update data with responses
            for idx, item in enumerate(batch):
                item[response_field] = responses[idx]

            yield batch  # Return processed batch
