import json


def process_json(input_file, output_file):
    # 读取所有行到列表中
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    result = []

    for line in lines:
        data = json.loads(line.strip())

        # 提取content字段（取messages数组的第一个元素）
        content = data["messages"][0]["content"] if data.get("messages") and len(data["messages"]) > 0 else ""

        # 处理images字段：取第一个元素
        image_path = data["images"][0] if data.get("images") and len(data["images"]) > 0 else ""

        # 创建新对象并添加到结果列表
        new_obj = {
            "id": data.get("id", ""),
            "images": image_path,
            "content": content
        }
        result.append(new_obj)

    # 将整个结果列表写入JSON文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 直接指定文件路径
    input_file = "input.jsonl"
    output_file = "output.json"

    print(f"开始处理文件: {input_file} -> {output_file}")
    process_json(input_file, output_file)
    print("文件处理完成！")
