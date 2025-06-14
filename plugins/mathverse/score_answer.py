# plugins/mathverse/score_answer.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import logging

from vllm import LLM, SamplingParams

from cores.batcher import DataBatcher
from cores.executor import PipelineExecutor
from cores.filter import DataFilter
from cores.loader import DataLoader
from cores.saver import DataSaver
from models.vllm import VLLMInferenceModel
from plugins.mathverse.processor import ScoreAnswerPreProcessor, CleanJudgeTagPostProcessor
from plugins.mathverse.prompts import demo_prompt_score

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
    if args.multimodal_input and not args.multimodal_output:
        logger.error("Output file must be specified for multimodal dataset")
        return False

    # Validate text-only parameters
    if args.text_only_input and not args.text_only_output:
        logger.error("Output file must be specified for text-only dataset")
        return False

    return True


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
    parser.add_argument("--index_field", type=str, default="sample_index",
                        help="Field for record identification")
    parser.add_argument("--query_field", type=str, default="extraction",
                        help="Field containing prompt queries")
    parser.add_argument("--question_field", type=str, default="question",
                        help="Field containing math questions")
    parser.add_argument("--answer_field", type=str, default="answer",
                        help="Field containing ground truths")
    parser.add_argument("--response_field", type=str, default="judgement",
                        help="Field to store generated responses")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Processing batch size")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun all samples regardless of status")

    args = parser.parse_args()

    # Validate arguments
    if not validate_arguments(args):
        return

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
        model_name=args.model_name
    )

    # Create shared components
    data_loader = DataLoader(args.index_field, args.response_field)
    data_filter = DataFilter(args.response_field, args.rerun)
    data_preprocessor = ScoreAnswerPreProcessor(args.query_field,
                                                args.question_field, args.answer_field,
                                                demo_prompt_score)
    data_postprocessor = CleanJudgeTagPostProcessor(args.response_field)
    data_batcher = DataBatcher(inference_model, data_preprocessor, data_postprocessor, batch_size=args.batch_size)

    # Process multimodal dataset
    if args.multimodal_input:
        logger.info("Processing multimodal dataset...")
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
