# generate_response.py
import argparse
import importlib.util
import logging
import os
import sys

from vllm import LLM, SamplingParams

from cores.batcher import DataBatcher
from cores.executor import PipelineExecutor
from cores.filter import DataFilter
from cores.loader import DataLoader
from cores.processor import MultimodalDataPreProcessor, TextOnlyDataPreProcessor
from cores.saver import DataSaver
from models.vllm import VLLMInferenceModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def validate_arguments(args) -> bool:
    """Validates command-line arguments"""
    # Verify at least one input is provided
    if not args.multimodal_input and not args.text_only_input:
        logger.error("At least one input file must be provided")
        return False

    # Validate multimodal parameters
    if args.multimodal_input:
        if not args.image_base_path:
            logger.error("Image base path is required for multimodal processing")
            return False
        if not args.multimodal_output:
            logger.error("Output file must be specified for multimodal dataset")
            return False

    # Validate text-only parameters
    if args.text_only_input and not args.text_only_output:
        logger.error("Output file must be specified for text-only dataset")
        return False

    # Validate system prompt file exists if specified
    if args.system_prompt_file and not os.path.isfile(args.system_prompt_file):
        logger.error(f"System prompt file not found: {args.system_prompt_file}")
        return False

    return True


def load_system_prompt(file_path: str, variable_name: str = "SYSTEM_PROMPT") -> str:
    """Loads a system prompt variable from a Python file"""
    # Create module specification
    spec = importlib.util.spec_from_file_location("prompt_module", file_path)
    if spec is None:
        raise ImportError(f"Could not import from file: {file_path}")

    # Create module and execute
    module = importlib.util.module_from_spec(spec)
    sys.modules["prompt_module"] = module
    spec.loader.exec_module(module)

    # Get system prompt variable
    if not hasattr(module, variable_name):
        raise AttributeError(f"Variable '{variable_name}' not found in {file_path}")

    return getattr(module, variable_name)


def main():
    parser = argparse.ArgumentParser(description="Generate responses using inference model")

    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model identifier name")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Tensor parallel configuration")
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Maximum model context length")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling value")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum tokens to generate")

    # System prompt parameters
    parser.add_argument("--system_prompt_file", type=str, default=None,
                        help="Python file containing the system prompt variable")
    parser.add_argument("--system_prompt_variable", type=str, default="SYSTEM_PROMPT",
                        help="Variable name containing the system prompt (default: SYSTEM_PROMPT)")

    # Dataset parameters
    parser.add_argument("--multimodal_input", type=str,
                        help="Input file for multimodal dataset")
    parser.add_argument("--multimodal_output", type=str,
                        help="Output file for multimodal results")
    parser.add_argument("--text_only_input", type=str,
                        help="Input file for text-only dataset")
    parser.add_argument("--text_only_output", type=str,
                        help="Output file for text-only results")

    # Processing parameters
    parser.add_argument("--image_base_path", type=str,
                        help="Base path for images (required for multimodal)")
    parser.add_argument("--index_field", type=str, default="sample_index",
                        help="Field for record identification")
    parser.add_argument("--query_field", type=str, default="query_cot",
                        help="Field containing prompt queries")
    parser.add_argument("--image_field", type=str, default="image",
                        help="Field containing image paths (multimodal)")
    parser.add_argument("--response_field", type=str, default="model_answer",
                        help="Field to store generated responses")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Processing batch size")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun all samples regardless of status")

    args = parser.parse_args()

    # Validate arguments
    if not validate_arguments(args):
        return

    # Load system prompt
    system_prompt = None
    if args.system_prompt_file:
        try:
            system_prompt = load_system_prompt(
                args.system_prompt_file,
                args.system_prompt_variable
            )
        except Exception as e:
            logger.error(f"Failed to load system prompt: {str(e)}")
            return
    else:
        logger.info("No system prompt file specified. Using default")

    # Initialize model
    logger.info(f"Loading model from: {args.model_path}")
    llm_instance = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )
    sampling_config = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    inference_model = VLLMInferenceModel(
        llm=llm_instance,
        sampling_params=sampling_config,
        model_name=args.model_name,
        system_prompt=system_prompt
    )

    # Create shared components
    data_loader = DataLoader(args.index_field, args.response_field)
    data_filter = DataFilter(args.response_field, args.rerun)

    # Process multimodal dataset
    if args.multimodal_input:
        logger.info("Processing multimodal dataset...")

        multimodal_preprocessor = MultimodalDataPreProcessor(
            args.query_field,
            args.image_base_path,
            args.image_field
        )
        data_batcher = DataBatcher(inference_model, multimodal_preprocessor, batch_size=args.batch_size)
        data_saver = DataSaver(args.multimodal_output)

        multimodal_pipeline = PipelineExecutor(
            data_loader,
            data_filter,
            data_batcher,
            data_saver
        )
        multimodal_pipeline.execute_pipeline(
            args.multimodal_input,
            args.multimodal_output,
            args.rerun
        )
    else:
        logger.info("Skipping multimodal dataset")

    # Process text-only dataset
    if args.text_only_input:
        logger.info("Processing text-only dataset...")

        text_preprocessor = TextOnlyDataPreProcessor(args.query_field)
        data_batcher = DataBatcher(inference_model, text_preprocessor, batch_size=args.batch_size)
        data_saver = DataSaver(args.text_only_output)

        text_pipeline = PipelineExecutor(
            data_loader,
            data_filter,
            data_batcher,
            data_saver
        )
        text_pipeline.execute_pipeline(
            args.text_only_input,
            args.text_only_output,
            args.rerun
        )
    else:
        logger.info("Skipping text-only dataset")


if __name__ == "__main__":
    main()
