import argparse
import json
import os
from collections import defaultdict


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to JSON file with judgement results')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save output JSON with statistics')
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

    # 准备输出数据结构
    output_stats = {
        "by_subject_subfield": {},
        "by_subject_total": {},
        "by_version": {},
        "overall": {}
    }

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

    # 按subject和subfield分组统计
    subject_stats = {}
    for subject, subfields in subject_subfield_dict.items():
        subject_stats[subject] = {"subfields": {}, "total": {}}
        subject_right = 0
        subject_total = 0

        for subfield, judgements in subfields.items():
            subfield_right = sum(judgements)
            subfield_total = len(judgements)
            subfield_acc = subfield_right / subfield_total if subfield_total > 0 else 0

            subject_right += subfield_right
            subject_total += subfield_total

            # 添加到输出结构
            subject_stats[subject]["subfields"][subfield] = {
                "correct": subfield_right,
                "total": subfield_total,
                "accuracy": subfield_acc
            }

        # 添加学科总计
        subject_acc = subject_right / subject_total if subject_total > 0 else 0
        subject_stats[subject]["total"] = {
            "correct": subject_right,
            "total": subject_total,
            "accuracy": subject_acc
        }

    # 按版本统计
    version_stats = {}
    for version, judgements in version_dict.items():
        version_right = sum(judgements)
        version_total = len(judgements)
        version_acc = version_right / version_total if version_total > 0 else 0

        version_stats[version] = {
            "correct": version_right,
            "total": version_total,
            "accuracy": version_acc
        }

    # 总体统计
    total_acc = total_right / total_count if total_count > 0 else 0
    overall_stats = {
        "correct": total_right,
        "total": total_count,
        "accuracy": total_acc
    }

    # 填充输出结构
    output_stats["by_subject_subfield"] = subject_stats
    output_stats["by_version"] = version_stats
    output_stats["overall"] = overall_stats

    # 保存到JSON文件
    save_json(output_stats, args.output_file)
    print(f"Statistics saved to {args.output_file}")

    # 打印结果到控制台
    print("\n" + "=" * 50)
    print("Accuracy by Subject and Subfield:")
    print("=" * 50)

    for subject, data in subject_stats.items():
        print(f"\n{subject}:")
        for subfield, stats in data["subfields"].items():
            print(f"  {subfield}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        total_stats = data["total"]
        print(f"  TOTAL: {total_stats['accuracy']:.4f} ({total_stats['correct']}/{total_stats['total']})")
        print("-" * 50)

    print("\n" + "=" * 50)
    print("Accuracy by Problem Version:")
    print("=" * 50)

    for version, stats in version_stats.items():
        print(f"{version}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    print("\n" + "=" * 50)
    print(f"OVERALL ACCURACY: {overall_stats['accuracy']:.4f} ({overall_stats['correct']}/{overall_stats['total']})")
    print("=" * 50)


if __name__ == '__main__':
    main()
