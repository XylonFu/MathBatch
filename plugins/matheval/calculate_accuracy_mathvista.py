import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validates command-line arguments"""
    # 验证至少提供了一个输入
    if not args.multimodal_input and not args.text_only_input:
        logger.error("At least one input file must be provided")
        return False

    # 如果同时提供了两种输入，则不需要分别提供输出，只需要提供一个输出即可
    if args.multimodal_input and args.text_only_input:
        if not args.combined_output:
            logger.error(
                "When both multimodal and text-only inputs are provided, a combined output file must be specified with --combined_output")
            return False
    else:
        # 验证多模态参数
        if args.multimodal_input and not args.multimodal_output:
            logger.error("Output file must be specified for multimodal dataset")
            return False
        # 验证纯文本参数
        if args.text_only_input and not args.text_only_output:
            logger.error("Output file must be specified for text-only dataset")
            return False

    return True


def read_json(file_path: str) -> List[Dict[str, Any]]:
    """读取JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        raise


def save_json(data: Any, file_path: str) -> None:
    """保存JSON文件"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {str(e)}")
        raise


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算数据集统计信息"""
    # 初始化各维度的统计字典
    stats = {
        "question_type": defaultdict(list),
        "answer_type": defaultdict(list),
        "language": defaultdict(list),
        "source": defaultdict(list),
        "category": defaultdict(list),
        "task": defaultdict(list),
        "context": defaultdict(list),
        "grade": defaultdict(list),
        "skills": defaultdict(list),
    }

    total_right = 0
    total_count = len(results)

    # 遍历所有结果进行统计
    for inst in results:
        # 获取各维度信息
        question_type = inst.get("question_type", "Unknown")
        answer_type = inst.get("answer_type", "Unknown")

        # 获取metadata信息
        meta = inst.get('metadata', {}) or {}
        language = meta.get('language', 'Unknown')
        source = meta.get('source', 'Unknown')
        category = meta.get('category', 'Unknown')
        task = meta.get('task', 'Unknown')
        context = meta.get('context', 'Unknown')
        grade = meta.get('grade', 'Unknown')
        skills = meta.get('skills', [])

        # 如果skills是字符串，转换为列表
        if isinstance(skills, str):
            skills = [skills]
        # 如果没有skills，添加一个"Unknown"标签
        if not skills:
            skills = ["Unknown"]

        judgement = inst.get('judgement', 0)

        # 统计到各维度
        stats["question_type"][question_type].append(judgement)
        stats["answer_type"][answer_type].append(judgement)
        stats["language"][language].append(judgement)
        stats["source"][source].append(judgement)
        stats["category"][category].append(judgement)
        stats["task"][task].append(judgement)
        stats["context"][context].append(judgement)
        stats["grade"][grade].append(judgement)

        # 对于skills，每个技能都要统计
        for skill in skills:
            stats["skills"][skill].append(judgement)

        # 统计总正确数
        if judgement == 1:
            total_right += 1

    return {
        "dimension_stats": stats,
        "total_right": total_right,
        "total_count": total_count
    }


def generate_report(results: List[Dict[str, Any]], dataset_type: str) -> Dict[str, Any]:
    """生成统计报告"""
    # 计算统计数据
    stats_data = calculate_statistics(results)
    dim_stats = stats_data["dimension_stats"]
    total_right = stats_data["total_right"]
    total_count = stats_data["total_count"]

    # 定义统计字典类型
    StatsDict = Dict[str, Union[int, float]]
    CategoryStats = Dict[str, StatsDict]

    # 准备输出数据结构
    output_stats = {
        "average": {
            "accuracy": total_right / total_count if total_count > 0 else 0,
            "correct": total_right,
            "total": total_count
        }
    }

    # 按维度计算统计信息
    dimensions = [
        "question_type", "answer_type", "language", "source",
        "category", "task", "context", "grade", "skills"
    ]

    for dim in dimensions:
        dim_dict = {}
        for category, judgements in dim_stats[dim].items():
            correct = sum(judgements)
            total = len(judgements)
            accuracy = correct / total if total > 0 else 0

            dim_dict[category] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }

        output_stats[dim] = dim_dict

    # 打印结果到控制台
    print(f"\n{dataset_type} Dataset Results")
    print("=" * 60)
    avg = output_stats["average"]
    print(f"OVERALL ACCURACY: {avg['accuracy']:.4f} ({avg['correct']}/{avg['total']})")

    # 打印各维度统计
    for dim in dimensions:
        print(f"\nAccuracy by {dim.replace('_', ' ').title()}:")
        dim_data = output_stats[dim]

        # 按正确率降序排序
        sorted_items = sorted(dim_data.items(), key=lambda x: x[1]["accuracy"], reverse=True)

        for category, stats in sorted_items:
            print(f"  {category}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    print("=" * 60)

    return output_stats


def process_dataset(input_file: str, output_file: str, dataset_type: str) -> None:
    """处理单个数据集"""
    logger.info(f"Processing {dataset_type} dataset from {input_file}")
    results = read_json(input_file)
    logger.info(f"Loaded {len(results)} records for {dataset_type} dataset")

    stats = generate_report(results, dataset_type)
    save_json(stats, output_file)


def main() -> None:
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

    # 合并输出参数（当同时提供两种输入时）
    parser.add_argument("--combined_output", type=str,
                        help="Output file for combined results (requires both multimodal_input and text_only_input)")

    args = parser.parse_args()

    # 验证参数
    if not validate_arguments(args):
        return

    # 如果同时提供了多模态和纯文本输入，则合并处理
    if args.multimodal_input and args.text_only_input:
        # 读取两个数据集
        multimodal_results = read_json(args.multimodal_input)
        text_only_results = read_json(args.text_only_input)

        # 合并列表
        combined_results = multimodal_results + text_only_results
        logger.info(
            f"Loaded {len(multimodal_results)} multimodal records and {len(text_only_results)} text-only records")
        logger.info(f"Total combined records: {len(combined_results)}")

        # 生成合并报告
        stats = generate_report(combined_results, "Combined")

        # 保存合并结果到指定的输出文件
        save_json(stats, args.combined_output)
        return

    # 否则分别处理各自的数据集
    if args.multimodal_input:
        process_dataset(
            args.multimodal_input,
            args.multimodal_output,
            "Multimodal"
        )

    if args.text_only_input:
        process_dataset(
            args.text_only_input,
            args.text_only_output,
            "Text-Only"
        )


if __name__ == '__main__':
    main()
