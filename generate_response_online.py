import argparse
import asyncio
import base64
import json
import mimetypes
import os

import aiofiles
from openai import AsyncOpenAI
from tqdm import tqdm


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
            except Exception:
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
        except Exception:
            if attempt < args.max_retries - 1:
                await asyncio.sleep(args.retry_delay)
            else:
                raise


async def get_openai_response(item, semaphore, args):
    async with semaphore:
        client = AsyncOpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.api_timeout
        )

        messages = await build_messages(item, args)
        if not messages:
            return None

        return await request_openai_response(messages, client, args)


async def process_item(item, semaphore, queue, args):
    try:
        response = await get_openai_response(item, semaphore, args)
        if response is not None:
            item[args.response_field] = response
        else:
            item["error"] = "No valid content"
    except Exception as e:
        item["error"] = str(e)

    await queue.put(item)


async def write_results(output_file, queue, total, write_batch_size):
    async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
        batch = []
        processed = 0

        with tqdm(total=total, desc="Processing", unit="item", mininterval=0.5) as progress:
            while processed < total:
                item = await queue.get()
                batch.append(item)

                if len(batch) >= write_batch_size:
                    await f.write("\n".join(json.dumps(i, ensure_ascii=False) for i in batch) + "\n")
                    progress.update(len(batch))
                    processed += len(batch)
                    batch = []

                queue.task_done()

            if batch:
                await f.write("\n".join(json.dumps(i, ensure_ascii=False) for i in batch) + "\n")
                progress.update(len(batch))


async def load_processed_ids(output_file, index_field, response_field):
    processed_ids = set()
    if os.path.exists(output_file):
        async with aiofiles.open(output_file, "r", encoding="utf-8") as f:
            async for line in f:
                try:
                    data = json.loads(line)
                    if data.get(index_field) and data.get(response_field):
                        processed_ids.add(str(data[index_field]))
                except:
                    continue
    return processed_ids


async def load_pending_items(input_file, processed_ids, args):
    items = []
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
            except:
                continue
    return items


async def execute_processing_tasks(items, semaphore, queue, args):
    tasks = [asyncio.create_task(process_item(item, semaphore, queue, args)) for item in items]
    await asyncio.gather(*tasks)


async def process_jsonl_file(args):
    concurrent_semaphore = asyncio.Semaphore(args.concurrent_batch_size)
    queue = asyncio.Queue()

    processed_ids = await load_processed_ids(args.output_file, args.index_field, args.response_field)
    items = await load_pending_items(args.input_file, processed_ids, args)

    if not items:
        return

    print(f"Processing {len(items)} items...")

    write_task = asyncio.create_task(write_results(args.output_file, queue, len(items), args.write_batch_size))
    await execute_processing_tasks(items, concurrent_semaphore, queue, args)
    await queue.join()
    await write_task


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--base_url', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--concurrent_batch_size', type=int, default=100)
    parser.add_argument('--write_batch_size', type=int, default=100)
    parser.add_argument('--api_timeout', type=int, default=120)
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--retry_delay', type=int, default=2)

    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--instructions', type=str, default=None)

    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    parser.add_argument('--index_field', type=str, default='id')
    parser.add_argument('--query_field', type=str, default='query')
    parser.add_argument('--response_field', type=str, default='response')

    parser.add_argument('--image_field', type=str, default=None)
    parser.add_argument('--image_base_path', type=str, default=None)

    return parser.parse_args()


async def main():
    args = parse_args()
    await process_jsonl_file(args)
    print("Processing completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
