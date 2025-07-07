# generate_response_online.py
import argparse
import asyncio
import json
import os

import aiofiles
from openai import AsyncOpenAI
from tqdm import tqdm


async def get_openai_response(query, semaphore, args):
    async with semaphore:
        client = AsyncOpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.api_timeout
        )

        for attempt in range(args.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": query}],
                )
                return response.choices[0].message.content
            except Exception:
                if attempt < args.max_retries - 1:
                    await asyncio.sleep(args.retry_delay * (attempt + 1))
                else:
                    raise


async def process_item(item, semaphore, queue, args):
    try:
        response = await get_openai_response(item[args.query_field], semaphore, args)
        item[args.response_field] = response
    except Exception as e:
        item["error"] = str(e)

    await queue.put(item)


async def write_results(output_file, queue, total):
    async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
        with tqdm(total=total, desc="Processing", unit="item") as progress:
            processed = 0
            while processed < total:
                item = await queue.get()
                await f.write(json.dumps(item, ensure_ascii=False) + "\n")
                await f.flush()
                progress.update(1)
                queue.task_done()
                processed += 1


async def process_jsonl_file(args):
    semaphore = asyncio.Semaphore(args.batch_size)
    queue = asyncio.Queue()

    processed_ids = set()
    if os.path.exists(args.output_file):
        async with aiofiles.open(args.output_file, "r", encoding="utf-8") as f:
            async for line in f:
                try:
                    data = json.loads(line)
                    if data.get(args.index_field) and data.get(args.response_field):
                        processed_ids.add(data[args.index_field])
                except json.JSONDecodeError:
                    continue

    items = []
    async with aiofiles.open(args.input_file, "r", encoding="utf-8") as f:
        async for line in f:
            try:
                item = json.loads(line)
                if (item.get(args.index_field) not in processed_ids and
                        item.get(args.query_field) and
                        not item.get(args.response_field)):
                    items.append(item)
            except json.JSONDecodeError:
                continue

    total_items = len(items)
    if total_items == 0:
        print("All items already processed")
        return

    write_task = asyncio.create_task(write_results(args.output_file, queue, total_items))

    processing_tasks = [
        asyncio.create_task(process_item(item, semaphore, queue, args))
        for item in items
    ]

    await asyncio.gather(*processing_tasks)
    await queue.join()
    await write_task


def parse_args():
    parser = argparse.ArgumentParser(description='Process JSONL files asynchronously')

    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--base_url', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--api_timeout', type=int, default=600)
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--retry_delay', type=int, default=5)

    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    parser.add_argument('--index_field', type=str, default="id")
    parser.add_argument('--query_field', type=str, default="query")
    parser.add_argument('--response_field', type=str, default="response")

    return parser.parse_args()


async def main():
    args = parse_args()
    await process_jsonl_file(args)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
