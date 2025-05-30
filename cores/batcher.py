# cores/batcher.py
import logging
from typing import List, Dict, Any, Optional

from models.base import BaseInferenceModel
from .base import BaseDataBatcher, BaseDataProcessor

logger = logging.getLogger(__name__)


class DataBatcher(BaseDataBatcher):
    """Concrete implementation of data batcher"""

    def __init__(self, model: BaseInferenceModel, data_preproceesor: BaseDataProcessor,
                 data_postprocessor: Optional[BaseDataProcessor] = None, batch_size: int = 500):
        self.model = model
        self.batch_size = batch_size
        self.data_preprocessor = data_preproceesor
        self.data_postprocessor = data_postprocessor

    def process_batches(
            self,
            data: List[Dict[str, Any]],
            response_field: str
    ) -> List[Dict[str, Any]]:
        """Processes data and generates responses"""
        # Process in batches
        for start_index in range(0, len(data), self.batch_size):
            batch = data[start_index:start_index + self.batch_size]

            # Prepare model inputs
            model_inputs = self.data_preprocessor.process_batch(batch)

            # Get model responses
            responses = self.model.generate_responses(
                model_inputs["prompts"],
                model_inputs["images"]
            )

            # Update data with responses
            for idx, item in enumerate(batch):
                item[response_field] = responses[idx]

            # Apply post-processing
            if self.data_postprocessor:
                batch = self.data_postprocessor.process_batch(batch)

            yield batch  # Return processed batch
