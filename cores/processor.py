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

        image_paths = item.get(self.image_field, [])
        images = []

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Process image if available
        for img_path in image_paths:
            if img_path and img_path.strip():
                full_path = os.path.join(self.image_base_path, img_path)
                if os.path.exists(full_path):
                    try:
                        images.append(Image.open(full_path).convert("RGB"))
                    except Exception as e:
                        logger.error(f"Image loading error: {full_path} - {str(e)}")

        if images:
            result["images"] = images

        return result

    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List]:
        """Processes a batch of data items"""
        prompts = []
        images = []

        for item in batch:
            processed = self.process_item(item)
            prompts.append(processed["prompt"])
            images.append(processed.get("images", []))

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


class StripPostProcessor(BaseDataProcessor):
    def __init__(self, response_field: str):
        super().__init__(field_name=response_field)
        self.response_field = response_field

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item[self.response_field] = item[self.response_field].strip()
        return item

    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List]:
        responses = [self.process_item(item)[self.response_field] for item in batch]
        return {"responses": responses}
