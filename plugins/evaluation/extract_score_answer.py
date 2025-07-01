import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import logging
from typing import List, Tuple

from vllm import LLM, SamplingParams

from cores.batcher import DataBatcher
from cores.executor import PipelineExecutor
from cores.filter import DataFilter
from cores.loader import DataLoader
from cores.saver import DataSaver
from models.vllm import VLLMInferenceModel
from plugins.evaluation.processor import (
    CleanExtractTagPostProcessor,
    ExtractionAnswerPreProcessor,
    ScoreAnswerPreProcessor,
    CleanJudgeTagPostProcessor
)
from plugins.evaluation.prompts import demo_prompt_extract, demo_prompt_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def validate_file_lists(inputs: List[str], intermediates: List[str], outputs: List[str], mode: str) -> bool:
    """Validate file list lengths and extensions"""
    if len(inputs) != len(intermediates) or len(inputs) != len(outputs):
        logger.error(f"Number of {mode} input, intermediate and output files must match")
        return False

    for path in inputs + intermediates + outputs:
        if not path.endswith('.json'):
            logger.error(f"All file paths must be .json files. Invalid: {path}")
            return False

    return True


def validate_arguments(args) -> bool:
    """Validates command-line arguments"""
    # Check at least one input is provided
    if not args.multimodal_inputs and not args.text_only_inputs:
        logger.error("At least one input file must be provided")
        return False

    # Validate multimodal parameters
    if args.multimodal_inputs:
        if not args.multimodal_intermediates or not args.multimodal_outputs:
            logger.error("Output files must be specified for multimodal datasets")
            return False
        if not validate_file_lists(
                args.multimodal_inputs,
                args.multimodal_intermediates,
                args.multimodal_outputs,
                "multimodal"
        ):
            return False

    # Validate text-only parameters
    if args.text_only_inputs:
        if not args.text_only_intermediates or not args.text_only_outputs:
            logger.error("Output files must be specified for text-only datasets")
            return False
        if not validate_file_lists(
                args.text_only_inputs,
                args.text_only_intermediates,
                args.text_only_outputs,
                "text-only"
        ):
            return False

    # Validate field names
    if not args.extract_query_field or not args.extract_response_field:
        logger.error("Extraction fields must be specified")
        return False
    if not args.score_query_field or not args.score_response_field:
        logger.error("Scoring fields must be specified")
        return False

    return True


def create_batchers(
        inference_model: VLLMInferenceModel,
        extract_query_field: str,
        extract_response_field: str,
        score_query_field: str,
        score_question_field: str,
        score_answer_field: str,
        score_response_field: str,
        batch_size: int
) -> Tuple[DataBatcher, DataBatcher]:
    """Create batchers for extraction and scoring phases"""
    # Extraction phase components
    extract_preprocessor = ExtractionAnswerPreProcessor(
        extract_query_field,
        demo_prompt_extract
    )
    extract_postprocessor = CleanExtractTagPostProcessor(extract_response_field)
    extract_batcher = DataBatcher(
        inference_model,
        extract_preprocessor,
        extract_postprocessor,
        batch_size=batch_size
    )

    # Scoring phase components
    score_preprocessor = ScoreAnswerPreProcessor(
        score_query_field,
        score_question_field,
        score_answer_field,
        demo_prompt_score
    )
    score_postprocessor = CleanJudgeTagPostProcessor(score_response_field)
    score_batcher = DataBatcher(
        inference_model,
        score_preprocessor,
        score_postprocessor,
        batch_size=batch_size
    )

    return extract_batcher, score_batcher


def process_files(
        inputs: List[str],
        intermediates: List[str],
        outputs: List[str],
        extract_batcher: DataBatcher,
        score_batcher: DataBatcher,
        index_field: str,
        extract_response_field: str,
        score_response_field: str,
        rerun: bool,
        mode: str
):
    """Process a set of files through extraction and scoring pipeline"""
    for input_path, intermediate_path, output_path in zip(inputs, intermediates, outputs):
        logger.info(f"Processing {mode} file: {input_path}")

        # Extraction phase
        logger.info(f"Starting extraction phase → {intermediate_path}")
        extract_loader = DataLoader(index_field, extract_response_field)
        extract_filter = DataFilter(extract_response_field, rerun)
        extract_saver = DataSaver(intermediate_path)
        extract_pipeline = PipelineExecutor(
            extract_loader,
            extract_filter,
            extract_batcher,
            extract_saver
        )
        extract_pipeline.execute_pipeline(input_path, intermediate_path, rerun)

        # Scoring phase
        logger.info(f"Starting scoring phase → {output_path}")
        score_loader = DataLoader(index_field, score_response_field)
        score_filter = DataFilter(score_response_field, rerun)
        score_saver = DataSaver(output_path)
        score_pipeline = PipelineExecutor(
            score_loader,
            score_filter,
            score_batcher,
            score_saver
        )
        score_pipeline.execute_pipeline(intermediate_path, output_path, rerun)

        logger.info(f"Completed processing for {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end answer extraction and scoring pipeline"
    )

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

    # File list parameters (supports multiple files)
    parser.add_argument("--multimodal_inputs", type=str, nargs='*', default=[],
                        help="Input files for multimodal datasets (.jsonl)")
    parser.add_argument("--multimodal_intermediates", type=str, nargs='*', default=[],
                        help="Intermediate files for multimodal extraction results (.jsonl)")
    parser.add_argument("--multimodal_outputs", type=str, nargs='*', default=[],
                        help="Output files for multimodal scoring results (.jsonl)")

    parser.add_argument("--text_only_inputs", type=str, nargs='*', default=[],
                        help="Input files for text-only datasets (.jsonl)")
    parser.add_argument("--text_only_intermediates", type=str, nargs='*', default=[],
                        help="Intermediate files for text-only extraction results (.jsonl)")
    parser.add_argument("--text_only_outputs", type=str, nargs='*', default=[],
                        help="Output files for text-only scoring results (.jsonl)")

    # Processing parameters
    parser.add_argument("--index_field", type=str, default="sample_index",
                        help="Field for record identification")
    parser.add_argument("--extract_query_field", type=str, default="model_answer",
                        help="Field containing prompt queries for extraction")
    parser.add_argument("--extract_response_field", type=str, default="extraction",
                        help="Field to store extraction results")
    parser.add_argument("--score_query_field", type=str, default="extraction",
                        help="Field containing queries for scoring")
    parser.add_argument("--score_question_field", type=str, default="question",
                        help="Field containing math questions")
    parser.add_argument("--score_answer_field", type=str, default="answer",
                        help="Field containing ground truths")
    parser.add_argument("--score_response_field", type=str, default="judgement",
                        help="Field to store scoring results")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Processing batch size")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun all samples regardless of status")

    args = parser.parse_args()

    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)

    # Initialize model (only once)
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

    # Create batchers for both phases
    extract_batcher, score_batcher = create_batchers(
        inference_model,
        args.extract_query_field,
        args.extract_response_field,
        args.score_query_field,
        args.score_question_field,
        args.score_answer_field,
        args.score_response_field,
        args.batch_size
    )

    # Process multimodal files
    if args.multimodal_inputs:
        logger.info(f"Processing {len(args.multimodal_inputs)} multimodal datasets")
        process_files(
            args.multimodal_inputs,
            args.multimodal_intermediates,
            args.multimodal_outputs,
            extract_batcher,
            score_batcher,
            args.index_field,
            args.extract_response_field,
            args.score_response_field,
            args.rerun,
            "multimodal"
        )

    # Process text-only files
    if args.text_only_inputs:
        logger.info(f"Processing {len(args.text_only_inputs)} text-only datasets")
        process_files(
            args.text_only_inputs,
            args.text_only_intermediates,
            args.text_only_outputs,
            extract_batcher,
            score_batcher,
            args.index_field,
            args.extract_response_field,
            args.score_response_field,
            args.rerun,
            "text-only"
        )

    logger.info("All processing complete")


if __name__ == "__main__":
    main()
