import argparse
import json


def process_jsonl(input_file, output_file):
    prefix_to_remove = "/gpfs/work/int/xinlongfu24/xinlong_fu/downloads/datasets/STAR/G-LLaVA/"

    with open(input_file, "r", encoding="utf-8") as infile, \
            open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            data = json.loads(line.strip())

            # 提取content字段（取messages数组的第一个元素）
            content = data["messages"][0]["content"] if data.get("messages") and len(data["messages"]) > 0 else ""

            # 处理images字段：取第一个元素并移除指定路径前缀
            image_path = ""
            if data.get("images") and len(data["images"]) > 0:
                image_path = data["images"][0].replace(prefix_to_remove, "", 1).replace("//", "/")

            # 创建新对象并写入
            new_obj = {
                "content": content,
                "images": image_path
            }
            outfile.write(json.dumps(new_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理JSONL文件")
    parser.add_argument("--input_file", required=True, help="输入文件路径")
    parser.add_argument("--output_file", required=True, help="输出文件路径")

    args = parser.parse_args()

    print(f"开始处理文件: {args.input_file} -> {args.output_file}")
    process_jsonl(args.input_file, args.output_file)
    print("文件处理完成！")
