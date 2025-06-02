import argparse
import json
import logging
import os
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_arguments(args) -> bool:
    """Validates command-line arguments"""
    # 验证至少提供了一个输入
    if not args.multimodal_input and not args.text_only_input:
        logger.error("At least one input file must be provided")
        return False

    # 验证多模态参数
    if args.multimodal_input and not args.multimodal_output:
        logger.error("Output file must be specified for multimodal dataset")
        return False

    # 验证纯文本参数
    if args.text_only_input and not args.text_only_output:
        logger.error("Output file must be specified for text-only dataset")
        return False

    return True


def read_json(file_path):
    """读取JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        raise


def save_json(data, file_path):
    """保存JSON文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {str(e)}")
        raise


def calculate_statistics(results):
    """计算数据集统计信息"""
    subject_subfield_dict = defaultdict(lambda: defaultdict(list))
    version_dict = defaultdict(list)
    total_right = 0
    total_count = len(results)

    # 遍历所有结果进行统计
    for inst in results:
        # 获取subject和subfield信息
        meta = inst.get('metadata', {}) or inst.get('category', {}) or {}
        subject = meta.get('subject', 'Unknown')
        subfield = meta.get('subfield', 'Unknown')
        version = inst.get('problem_version', 'Unknown')
        judgement = inst.get('judgement', 0)

        # 统计到对应分类
        subject_subfield_dict[subject][subfield].append(judgement)
        version_dict[version].append(judgement)

        # 统计总正确数
        if judgement == 1:
            total_right += 1

    return subject_subfield_dict, version_dict, total_right, total_count


def generate_report(results, dataset_type):
    """生成统计报告"""
    subject_subfield_dict, version_dict, total_right, total_count = calculate_statistics(results)

    # 准备输出数据结构
    output_stats = {
        "by_subject_subfield": {},
        "by_subject_total": {},
        "by_version": {},
        "overall": {}
    }

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

    # 打印结果到控制台
    print(f"\n{dataset_type} Dataset Results")
    print("=" * 60)
    print(f"OVERALL ACCURACY: {overall_stats['accuracy']:.4f} ({overall_stats['correct']}/{overall_stats['total']})")

    print("\nAccuracy by Subject:")
    for subject, data in subject_stats.items():
        total_stats = data["total"]
        print(f"  {subject}: {total_stats['accuracy']:.4f} ({total_stats['correct']}/{total_stats['total']})")

    print("\nAccuracy by Problem Version:")
    for version, stats in version_stats.items():
        print(f"  {version}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    print("=" * 60)

    return output_stats


def process_dataset(input_file, output_file, dataset_type):
    """处理单个数据集"""
    logger.info(f"Processing {dataset_type} dataset from {input_file}")
    results = read_json(input_file)
    logger.info(f"Loaded {len(results)} records for {dataset_type} dataset")

    stats = generate_report(results, dataset_type)
    save_json(stats, output_file)


def main():
    parser = argparse.ArgumentParser(description="Generate statistics from judgement results")

    # 多模态数据集参数
    parser.add_argument("--multimodal_input", type=str,
                        help="Input file for multimodal dataset")
    parser.add_argument("--multimodal_output", type=str,
                        help="Output file for multimodal results")

    # 纯文本数据集参数
    parser.add_argument("--text_only_input", type=str,
                        help="Input file for text-only dataset")
    parser.add_argument("--text_only_output", type=str,
                        help="Output file for text-only results")

    args = parser.parse_args()

    # 验证参数
    if not validate_arguments(args):
        return

    # 处理多模态数据集
    if args.multimodal_input:
        process_dataset(
            args.multimodal_input,
            args.multimodal_output,
            "Multimodal"
        )

    # 处理纯文本数据集
    if args.text_only_input:
        process_dataset(
            args.text_only_input,
            args.text_only_output,
            "Text-Only"
        )


if __name__ == '__main__':
    main()
