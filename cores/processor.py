# cores/processor.py
import logging
import os
from typing import List, Dict, Any

from PIL import Image

from .base import BaseDataProcessor

logger = logging.getLogger(__name__)


class MultimodalDataPreProcessor(BaseDataProcessor):
    """Processor for multimodal (text + image) data"""

    def __init__(self, query_field: str, image_base_path: str, image_field: str):
        super().__init__(field_name=query_field)
        self.query_field = query_field
        self.image_base_path = image_base_path
        self.image_field = image_field

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single multimodal item"""
        result = {"prompt": item.get(self.query_field, "")}

        # Process image if available
        img_path = item.get(self.image_field, "")
        if img_path and img_path.strip():
            full_path = os.path.join(self.image_base_path, img_path)
            if os.path.exists(full_path):
                try:
                    result["image"] = Image.open(full_path).convert("RGB")
                except Exception as e:
                    logger.error(f"Image loading error: {full_path} - {str(e)}")
        return result

    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List]:
        """Processes a batch of data items"""
        prompts = []
        images = []

        for item in batch:
            processed = self.process_item(item)
            prompts.append(processed["prompt"])
            images.append(processed.get("image"))

        return {"prompts": prompts, "images": images}


class TextOnlyDataPreProcessor(BaseDataProcessor):
    """Processor for text-only data"""

    def __init__(self, query_field: str):
        super().__init__(field_name=query_field)
        self.query_field = query_field

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single text-only item"""
        return {"prompt": item[self.query_field]}

    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List]:
        """Processes a batch of data items"""
        prompts = [self.process_item(item)["prompt"] for item in batch]
        return {"prompts": prompts, "images": [None] * len(batch)}
