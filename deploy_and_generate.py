# deploy_and_generate.py
import argparse
import asyncio
import logging
from typing import Dict, Any

from generate_response_online import main as generate_responses_main
from generate_response_online import parse_args as parse_generate_args
from models.api_server import (
    start_vllm_server,
    start_lmdeploy_server,
    wait_server,
    stop_server
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def deploy_and_generate(args: argparse.Namespace) -> None:
    server_process = None

    try:
        if args.server_type == "vllm":
            logger.info(f"Starting vLLM server for model: {args.model_path}")
            server_process = start_vllm_server(
                conda_env_path=args.conda_env_path,
                model_path=args.model_path,
                served_model_name=args.served_model_name,
                devices=args.devices,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                host=args.host,
                port=args.port,
                api_key=args.api_key,
                chat_template=args.chat_template
            )
        elif args.server_type == "lmdeploy":
            logger.info(f"Starting LMDeploy server for model: {args.model_path}")
            server_process = start_lmdeploy_server(
                conda_env_path=args.conda_env_path,
                model_path=args.model_path,
                served_model_name=args.served_model_name,
                devices=args.devices,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                host=args.host,
                port=args.port,
                api_key=args.api_key,
                chat_template=args.chat_template
            )
        else:
            raise ValueError(f"Unsupported server type: {args.server_type}")

        logger.info(f"Waiting for server to start on {args.host}:{args.port}...")
        wait_server(host=args.host, port=args.port, timeout=600)
        logger.info("Server is ready!")

        arg_dict: Dict[str, Any] = {
            "api_key": args.api_key,
            "base_url": f"http://{args.host}:{args.port}/v1",
            "model": args.served_model_name,
            "input_file": args.input_file,
            "output_file": args.output_file,
            "query_field": args.query_field,
            "response_field": args.response_field,
            "index_field": args.index_field,
            "concurrent_batch_size": args.concurrent_batch_size,
            "write_batch_size": args.write_batch_size,
            "api_timeout": args.api_timeout,
            "max_retries": args.max_retries,
            "retry_delay": args.retry_delay,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }

        if args.instructions:
            arg_dict["instructions"] = args.instructions
        if args.image_field:
            arg_dict["image_field"] = args.image_field
        if args.image_base_path:
            arg_dict["image_base_path"] = args.image_base_path

        generate_args = parse_generate_args(arg_dict)

        logger.info(f"Starting generation process for {len(args.devices)} GPU(s)")
        await generate_responses_main(generate_args)
        logger.info("Generation completed successfully")

    except Exception as e:
        logger.error(f"Error during deployment/generation: {str(e)}")
        raise
    finally:
        if server_process is not None:
            logger.info("Stopping server...")
            try:
                stop_server(server_process, devices=args.devices)
                logger.info("Server stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping server: {str(e)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument('--server_type', type=str, required=True, choices=['vllm', 'lmdeploy'])
    server_group.add_argument('--conda_env_path', type=str, required=True)
    server_group.add_argument('--model_path', type=str, required=True)
    server_group.add_argument('--served_model_name', type=str, required=True)
    server_group.add_argument('--devices', type=int, nargs='+', default=[0])
    server_group.add_argument('--tensor_parallel_size', type=int, default=1)
    server_group.add_argument('--max_model_len', type=int, default=16384)
    server_group.add_argument('--max_num_seqs', type=int, default=512)
    server_group.add_argument('--host', type=str, default="127.0.0.1")
    server_group.add_argument('--port', type=int, default=8000)
    server_group.add_argument('--api_key', type=str, default="EMPTY")
    server_group.add_argument('--chat_template', type=str, default=None)

    generate_group = parser.add_argument_group('Generate Configuration')
    generate_group.add_argument('--input_file', type=str, required=True)
    generate_group.add_argument('--output_file', type=str, required=True)
    generate_group.add_argument('--index_field', type=str, default='id')
    generate_group.add_argument('--query_field', type=str, default='query')
    generate_group.add_argument('--response_field', type=str, default='response')

    image_group = parser.add_argument_group('Image Configuration')
    image_group.add_argument('--image_field', type=str, default=None)
    image_group.add_argument('--image_base_path', type=str, default=None)
    image_group.add_argument('--instructions', type=str, default=None)

    perf_group = parser.add_argument_group('Performance Configuration')
    perf_group.add_argument('--concurrent_batch_size', type=int, default=100)
    perf_group.add_argument('--write_batch_size', type=int, default=100)
    perf_group.add_argument('--api_timeout', type=int, default=600)
    perf_group.add_argument('--max_retries', type=int, default=3)
    perf_group.add_argument('--retry_delay', type=int, default=2)
    perf_group.add_argument('--temperature', type=float, default=0.0)
    perf_group.add_argument('--top_p', type=float, default=1.0)
    perf_group.add_argument('--max_tokens', type=int, default=4096)

    return parser.parse_args()


async def main() -> None:
    try:
        args = parse_args()
        await deploy_and_generate(args)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
