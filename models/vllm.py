# models/vllm.py
import base64
import logging
from io import BytesIO
from typing import List, Optional, Union, Dict

from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from .base import BaseInferenceModel

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing operations"""
    MIN_IMAGE_DIMENSION = 224

    @staticmethod
    def ensure_min_size(image: Image.Image, min_dim: int = MIN_IMAGE_DIMENSION) -> Image.Image:
        """Ensures image meets minimum dimension requirements"""
        width, height = image.size
        if min(width, height) < min_dim:
            scale = min_dim / min(width, height)
            return image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        return image

    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """Encodes image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class MessageConstructor:
    """Constructs model input messages"""
    DEFAULT_SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant."}

    def __init__(self, model_name: str, system_prompt: Optional[Dict[str, str]] = None):
        self.model_name = model_name
        self.is_internvl = "internvl" in model_name.lower()
        if not self.is_internvl:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def construct(
            self,
            prompt: str,
            image: Optional[Image.Image] = None
    ) -> Dict[str, Union[str, dict, List[dict]]]:
        """
        Constructs model input message
        :param prompt: Text prompt
        :param image: PIL image object
        :return: Dictionary containing prompt and optional multi_modal_data
        """
        messages = self._format_messages(prompt, image)

        if self.is_internvl:
            # For InternVL, return the raw messages list
            return {"messages": messages}
        else:
            chat_prompt = self._generate_chat_prompt(messages)
            entry = {"prompt": chat_prompt}
            if image:
                entry["multi_modal_data"] = {"image": image}
            return entry

    def _format_messages(
            self,
            prompt: str,
            image: Optional[Image.Image]
    ) -> List[dict]:
        """Formats message list with proper role and content"""
        user_content = self._build_user_content(prompt, image)
        return [
            self.system_prompt,
            {"role": "user", "content": user_content}
        ]

    def _build_user_content(
            self,
            prompt: str,
            image: Optional[Image.Image]
    ) -> Union[str, list]:
        """Builds user content section"""
        if image:
            processed_image = ImagePreprocessor.ensure_min_size(image)
            img_str = ImagePreprocessor.encode_image(processed_image)

            return [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                },
                {"type": "text", "text": prompt}
            ]
        return prompt

    def _generate_chat_prompt(self, messages: List[dict]) -> str:
        """Generates chat prompt template"""
        extra_args = {}
        if "qwen3" in self.model_name.lower():
            extra_args["enable_thinking"] = False

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **extra_args
        )


class VLLMInferenceModel(BaseInferenceModel):
    """vLLM implementation of the model inference interface"""

    def __init__(
            self,
            llm: LLM,
            sampling_params: SamplingParams,
            model_name: str,
            system_prompt: Optional[Dict[str, str]] = None
    ):
        self.llm = llm
        self.sampling_params = sampling_params
        self.model_name = model_name
        self.is_internvl = "internvl" in model_name.lower()
        self.message_constructor = MessageConstructor(model_name, system_prompt)

    def generate_responses(
            self,
            prompts: List[str],
            images: List[Optional[Image.Image]]
    ) -> List[str]:
        """Processes batches of prompts and images"""
        try:
            # Build model input requests
            requests = [
                self.message_constructor.construct(prompt, image)
                for prompt, image in zip(prompts, images)
            ]

            print(f"First request: {requests[0]}")

            # Generate responses
            if self.is_internvl:
                # For InternVL, use chat interface with messages
                outputs = self.llm.chat(
                    requests,
                    sampling_params=self.sampling_params
                )
                return [output.outputs[0].text for output in outputs]
            else:
                # For other models, use generate interface with prompts
                outputs = self.llm.generate(
                    requests,
                    sampling_params=self.sampling_params
                )
                return [output.outputs[0].text for output in outputs]

            print(f"First output: {outputs[0].outputs[0].text}")

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}", exc_info=True)
            return [""] * len(prompts)

    def generate_single_response(
            self,
            prompt: str,
            image: Optional[Image.Image] = None
    ) -> str:
        """Processes a single prompt"""
        return self.generate_responses([prompt], [image])[0]
