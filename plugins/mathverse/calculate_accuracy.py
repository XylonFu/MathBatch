import argparse
import json
from collections import defaultdict


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to JSON file with judgement results')
    args = parser.parse_args()

    # 读取JSON文件
    print(f"Reading {args.input_file}...")
    results = read_json(args.input_file)
    print(f"Loaded {len(results)} records")

    # 初始化统计字典
    subject_subfield_dict = defaultdict(lambda: defaultdict(list))
    version_dict = defaultdict(list)
    total_right = 0
    total_count = len(results)

    # 遍历所有结果进行统计
    for inst in results:
        # 获取subject和subfield信息（优先从metadata获取）
        meta = inst.get('metadata', {}) or inst.get('category', {})
        subject = meta.get('subject', 'Unknown')
        subfield = meta.get('subfield', 'Unknown')

        # 获取problem_version
        version = inst.get('problem_version', 'Unknown')

        # 获取judgement结果
        judgement = inst.get('judgement', 0)

        # 统计到对应分类
        subject_subfield_dict[subject][subfield].append(judgement)
        version_dict[version].append(judgement)

        # 统计总正确数
        if judgement == 1:
            total_right += 1

    # 打印按subject和subfield分组的准确率
    print("\n" + "=" * 50)
    print("Accuracy by Subject and Subfield:")
    print("=" * 50)

    for subject, subfields in subject_subfield_dict.items():
        subject_right = 0
        subject_total = 0

        for subfield, judgements in subfields.items():
            subfield_right = sum(judgements)
            subfield_total = len(judgements)
            subfield_acc = subfield_right / subfield_total if subfield_total > 0 else 0

            subject_right += subfield_right
            subject_total += subfield_total

            print(f"{subject} - {subfield}: {subfield_acc:.4f} ({subfield_right}/{subfield_total})")

        subject_acc = subject_right / subject_total if subject_total > 0 else 0
        print(f"{subject} TOTAL: {subject_acc:.4f} ({subject_right}/{subject_total})")
        print("-" * 50)

    # 打印按problem_version分组的准确率
    print("\n" + "=" * 50)
    print("Accuracy by Problem Version:")
    print("=" * 50)

    for version, judgements in version_dict.items():
        version_right = sum(judgements)
        version_total = len(judgements)
        version_acc = version_right / version_total if version_total > 0 else 0

        print(f"{version}: {version_acc:.4f} ({version_right}/{version_total})")

    # 打印总体准确率
    total_acc = total_right / total_count if total_count > 0 else 0
    print("\n" + "=" * 50)
    print(f"OVERALL ACCURACY: {total_acc:.4f} ({total_right}/{total_count})")
    print("=" * 50)


if __name__ == '__main__':
    main()
