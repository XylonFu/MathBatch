import json
import re


def extract_markdown_content(response):
    # 使用正则表达式提取 ```markdown ``` 内的内容
    pattern = r'```markdown(.*?)```'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    # 移除所有 <image> 标签
    content = content.replace('<image>', '').strip()
    return content


def extract_system_content(content_text):
    # 提取 "system:" 到 "student_alpha:" 之间的内容（不包括标识符）
    system_start = content_text.find('system:')
    if system_start == -1:
        return None

    # 找到system:之后内容的起始位置（跳过标识符本身）
    content_start = system_start + len('system:')

    # 找到student_alpha:标识符的位置
    student_start = content_text.find('student_alpha:', content_start)
    if student_start == -1:
        return None

    # 提取system:和student_alpha:之间的内容（不包括标识符）
    return content_text[content_start:student_start].strip()


def process_item(item):
    try:
        # 处理response字段
        response_content = extract_markdown_content(item['response'])
        if response_content is None:
            return None

        # 处理content字段
        system_content = extract_system_content(item['content'])
        if system_content is None:
            return None

        # 拼接内容
        final_content = system_content + '\n' + response_content

        # 处理images字段
        images = item['images']
        if isinstance(images, str):
            images = [images]
        elif isinstance(images, list) and images:
            # 确保images是包含单个字符串的列表
            images = [images[0]] if images else []
        else:
            return None

        return {
            "id": item["id"],
            "images": images,
            "messages": [
                {
                    "role": "assistant",
                    "content": final_content
                }
            ]
        }
    except (KeyError, TypeError, IndexError):
        return None


def main(input_file, output_file):
    # 读取输入JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化计数器
    total_count = len(data)
    skip_count = 0

    # 处理并写入JSONL
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data:
            processed = process_item(item)
            if processed is None:
                skip_count += 1
                continue

            f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')

    # 打印处理结果统计
    success_count = total_count - skip_count
    print(f"处理完成！共处理 {total_count} 条记录")
    print(f"成功处理: {success_count} 条")
    print(f"跳过处理: {skip_count} 条")


if __name__ == '__main__':
    input_filename = 'input.json'
    output_filename = 'output.jsonl'
    main(input_filename, output_filename)
