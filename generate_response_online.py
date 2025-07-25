# generate_response_online.py
import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import sys

import aiofiles
from openai import AsyncOpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


async def build_messages(item, args):
    content_list = []

    if args.query_field in item and item[args.query_field]:
        content_list.append({"type": "text", "text": item[args.query_field]})

    if args.image_field and args.image_field in item and item[args.image_field]:
        image_paths = item[args.image_field]
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        for img_path in image_paths:
            if not img_path:
                continue

            if img_path.startswith(("http://", "https://")):
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": img_path}
                })
                continue

            full_path = os.path.join(args.image_base_path, img_path) if args.image_base_path else img_path

            try:
                async with aiofiles.open(full_path, "rb") as image_file:
                    image_data = await image_file.read()

                mime_type, _ = mimetypes.guess_type(full_path)
                if not mime_type:
                    mime_type = "image/jpeg"

                b64_image = base64.b64encode(image_data).decode("utf-8")
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}"
                    }
                })
            except Exception as e:
                logger.error(f"Image loading error: {img_path} - {str(e)}")
                continue

    messages = []
    if args.instructions:
        messages.append({"role": "system", "content": args.instructions})

    if content_list:
        messages.append({"role": "user", "content": content_list})

    return messages if messages else None


async def request_openai_response(messages, client, args):
    for attempt in range(args.max_retries):
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < args.max_retries - 1:
                logger.warning(f"API attempt {attempt + 1}/{args.max_retries}: {str(e)}")
                await asyncio.sleep(args.retry_delay)
            else:
                logger.error(f"API failed after {args.max_retries} attempts: {str(e)}")
                raise


async def get_openai_response(item, semaphore, client, args):
    async with semaphore:
        messages = await build_messages(item, args)
        if not messages:
            return None

        return await request_openai_response(messages, client, args)


async def process_item(item, semaphore, client, queue, args):
    try:
        response = await get_openai_response(item, semaphore, client, args)
        if response:
            item[args.response_field] = response
        else:
            item["error"] = "No valid content"
    except Exception as e:
        item["error"] = str(e)
        logger.error(f"Processing error: {item.get(args.index_field, '')} - {str(e)}")

    await queue.put(item)


async def write_results(output_file, queue, total, write_batch_size):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
        batch = []
        processed = 0

        with tqdm(total=total, desc="Processing", unit="item", mininterval=0.5, file=sys.stderr) as progress:
            while True:
                item = await queue.get()

                if item is None:
                    queue.task_done()
                    break

                batch.append(item)
                queue.task_done()

                if len(batch) >= write_batch_size:
                    await f.write("\n".join(json.dumps(i, ensure_ascii=False) for i in batch) + "\n")
                    await f.flush()
                    progress.update(len(batch))
                    processed += len(batch)
                    batch = []

            if batch:
                await f.write("\n".join(json.dumps(i, ensure_ascii=False) for i in batch) + "\n")
                await f.flush()
                progress.update(len(batch))
                processed += len(batch)


async def load_processed_ids(output_file, index_field, response_field):
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            async with aiofiles.open(output_file, "r", encoding="utf-8") as f:
                async for line in f:
                    try:
                        data = json.loads(line)
                        index_value = data.get(index_field)

                        if index_value and data.get(response_field):
                            processed_ids.add(str(index_value))
                            continue

                        error_value = data.get("error")
                        if index_value and error_value:
                            if re.search(r'Error code: 400\b', error_value):
                                processed_ids.add(str(index_value))
                                logger.info(f"Skipping item {index_value} due to 400 error")

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {str(e)}")
        except Exception as e:
            logger.error(f"Output file error: {str(e)}")
    return processed_ids


async def load_pending_items(input_file, processed_ids, args):
    items = []
    try:
        async with aiofiles.open(input_file, "r", encoding="utf-8") as f:
            async for line in f:
                try:
                    item = json.loads(line)
                    item_id = str(item.get(args.index_field, ''))

                    has_content = bool(
                        (args.query_field in item and item[args.query_field]) or
                        (args.image_field and args.image_field in item and item[args.image_field])
                    )

                    if item_id in processed_ids:
                        continue
                    if not has_content:
                        continue
                    if item.get(args.response_field):
                        continue

                    items.append(item)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
    except Exception as e:
        logger.error(f"Input file error: {str(e)}")
    return items


async def execute_processing_tasks(items, semaphore, client, queue, args):
    tasks = [asyncio.create_task(process_item(item, semaphore, client, queue, args)) for item in items]
    await asyncio.gather(*tasks)


async def process_jsonl_file(args):
    concurrent_semaphore = asyncio.Semaphore(args.concurrent_batch_size)
    queue = asyncio.Queue()

    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.api_timeout
    )

    processed_ids = await load_processed_ids(args.output_file, args.index_field, args.response_field)
    items = await load_pending_items(args.input_file, processed_ids, args)

    if not items:
        logger.info("No items to process")
        return

    logger.info(f"Skipping {len(processed_ids)} items")
    logger.info(f"Processing {len(items)} items")

    write_task = asyncio.create_task(write_results(args.output_file, queue, len(items), args.write_batch_size))
    await execute_processing_tasks(items, concurrent_semaphore, client, queue, args)
    await queue.put(None)
    await queue.join()
    await write_task
    await client.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    api_group = parser.add_argument_group('API Configuration')
    api_group.add_argument('--api_key', type=str, required=True)
    api_group.add_argument('--base_url', type=str, required=True)
    api_group.add_argument('--model', type=str, required=True)

    perf_group = parser.add_argument_group('Performance Configuration')
    perf_group.add_argument('--concurrent_batch_size', type=int, default=100)
    perf_group.add_argument('--write_batch_size', type=int, default=100)
    perf_group.add_argument('--api_timeout', type=int, default=600)
    perf_group.add_argument('--max_retries', type=int, default=3)
    perf_group.add_argument('--retry_delay', type=int, default=2)

    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument('--temperature', type=float, default=0.0)
    model_group.add_argument('--top_p', type=float, default=1.0)
    model_group.add_argument('--max_tokens', type=int, default=4096)
    model_group.add_argument('--instructions', type=str, default=None)

    file_group = parser.add_argument_group('File Configuration')
    file_group.add_argument('--input_file', type=str, required=True)
    file_group.add_argument('--output_file', type=str, required=True)

    field_group = parser.add_argument_group('Field Configuration')
    field_group.add_argument('--index_field', type=str, default='id')
    field_group.add_argument('--query_field', type=str, default='query')
    field_group.add_argument('--response_field', type=str, default='response')

    image_group = parser.add_argument_group('Image Configuration')
    image_group.add_argument('--image_field', type=str, default=None)
    image_group.add_argument('--image_base_path', type=str, default=None)

    if args is None:
        return parser.parse_args()
    elif isinstance(args, dict):
        arg_list = []
        for key, value in args.items():
            if value is not None:
                arg_list.append(f"--{key.replace('_', '-')}")
                if not isinstance(value, bool):
                    arg_list.append(str(value))
        return parser.parse_args(arg_list)
    else:
        raise ValueError("args parameter must be None or a dictionary")


async def main(args=None):
    if args is None:
        args = parse_args()
    await process_jsonl_file(args)
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
