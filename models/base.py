# models/base.py
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class BaseInferenceModel(ABC):
    """Abstract base class for model inference interfaces"""

    @abstractmethod
    def generate_responses(
            self,
            prompts: List[str],
            images: List[Optional[List[Image.Image]]]
    ) -> List[str]:
        """Generates responses for batches of prompts and images"""
        pass

    @abstractmethod
    def generate_single_response(
            self,
            prompt: str,
            image: Optional[List[Image.Image]] = None
    ) -> str:
        """Generates response for a single prompt"""
        pass
