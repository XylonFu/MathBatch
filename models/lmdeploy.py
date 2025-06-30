# models/lmdeploy.py
import base64
import logging
from io import BytesIO
from typing import List, Optional, Dict

from PIL import Image

from lmdeploy import pipeline, GenerationConfig
from .base import BaseInferenceModel

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    @staticmethod
    def encode_image(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class MessageConstructor:
    DEFAULT_SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant."}

    def __init__(self, model_name: str, system_prompt: Optional[Dict[str, str]] = None):
        self.model_name = model_name
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def construct(
            self,
            prompt: str,
            images: Optional[List[Image.Image]] = None
    ) -> List[Dict]:
        user_content = []

        if images:
            for image in images:
                img_str = ImagePreprocessor.encode_image(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                })

        user_content.append({"type": "text", "text": prompt})

        return [
            self.system_prompt,
            {"role": "user", "content": user_content}
        ]


class LMDeployInferenceModel(BaseInferenceModel):

    def __init__(
            self,
            pipe: pipeline,
            gen_config: GenerationConfig,
            model_name: str,
            system_prompt: Optional[Dict[str, str]] = None
    ):
        self.pipe = pipe
        self.gen_config = gen_config
        self.model_name = model_name
        self.message_constructor = MessageConstructor(model_name, system_prompt)

    def generate_responses(
            self,
            prompts: List[str],
            images: List[Optional[List[Image.Image]]]
    ) -> List[str]:
        try:
            requests = [
                self.message_constructor.construct(prompt, image)
                for prompt, image in zip(prompts, images)
            ]

            print(f"First request: {requests[0]}")

            outputs = self.pipe(requests, gen_config=self.gen_config)

            print(f"First output: {outputs[0].text}")

            return [output.text for output in outputs]
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}", exc_info=True)
            return [""] * len(prompts)

    def generate_single_response(
            self,
            prompt: str,
            image: Optional[List[Image.Image]] = None
    ) -> str:
        return self.generate_responses([prompt], [image])[0]
