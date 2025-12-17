#!/usr/bin/env python3
"""
数据集完整性检查脚本
检查 ref_mask, ref_image, tar_image, tar_mask 四个文件夹的文件是否对应存在
"""

import os
import sys
from pathlib import Path


def check_dataset(data_dir: str) -> dict:
    """
    检查单个数据集目录的完整性

    Args:
        data_dir: 数据集目录路径 (包含 ref_mask, ref_image, tar_image, tar_mask 子目录)

    Returns:
        dict: 检查结果统计
    """
    data_path = Path(data_dir)

    # 需要检查的子目录
    required_dirs = ["ref_mask", "ref_image", "tar_image", "tar_mask"]

    results = {
        "total_files": 0,
        "missing_files": [],
        "invalid_paths": [],
        "ok_count": 0,
    }

    # 检查目录是否存在
    for d in required_dirs:
        dir_path = data_path / d
        if not dir_path.exists():
            print(f"[ERROR] 目录不存在: {dir_path}")
            results["invalid_paths"].append(str(dir_path))

    if results["invalid_paths"]:
        return results
    
    # 获取 ref_mask 中的所有文件作为基准
    ref_mask_dir = data_path / "ref_mask"
    mask_files = list(ref_mask_dir.iterdir())
    results["total_files"] = len(mask_files)

    print(f"\n检查目录: {data_dir}")
    print(f"共 {len(mask_files)} 个样本需要检查...\n")

    for mask_file in mask_files:
        filename = mask_file.name

        # 构建对应的其他文件路径
        ref_image_path = data_path / "ref_image" / filename
        tar_image_path = data_path / "tar_image" / filename
        tar_mask_path = data_path / "tar_mask" / filename

        missing = []

        if not ref_image_path.exists():
            missing.append(f"ref_image/{filename}")
        if not tar_image_path.exists():
            missing.append(f"tar_image/{filename}")
        if not tar_mask_path.exists():
            missing.append(f"tar_mask/{filename}")

        if missing:
            results["missing_files"].append(
                {"ref_mask": str(mask_file), "missing": missing}
            )
        else:
            results["ok_count"] += 1

    return results


def print_results(results: dict, data_dir: str):
    """打印检查结果"""
    print("=" * 60)
    print(f"数据集: {data_dir}")
    print("=" * 60)
    print(f"总文件数: {results['total_files']}")
    print(f"完整样本: {results['ok_count']}")
    print(f"缺失样本: {len(results['missing_files'])}")

    if results["missing_files"]:
        print("\n缺失详情 (前10个):")
        for item in results["missing_files"][:10]:
            print(f"  基准文件: {item['ref_mask']}")
            for m in item["missing"]:
                print(f"    ❌ 缺失: {m}")

        if len(results["missing_files"]) > 10:
            print(f"  ... 还有 {len(results['missing_files']) - 10} 个缺失样本")
    else:
        print("\n✅ 所有文件完整!")

    print()


def main():
    """主函数"""
    # 默认检查的数据集目录
    default_dirs = [
        "data/train/person",
        "data/train/person_head",
        "data/train/person_hair",
    ]

    # 如果命令行指定了目录，使用命令行参数
    if len(sys.argv) > 1:
        dirs_to_check = sys.argv[1:]
    else:
        dirs_to_check = default_dirs

    all_ok = True

    for data_dir in dirs_to_check:
        if not os.path.exists(data_dir):
            print(f"[SKIP] 目录不存在: {data_dir}")
            continue

        results = check_dataset(data_dir)
        print_results(results, data_dir)

        if results["missing_files"]:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("✅ 所有数据集检查通过!")
    else:
        print("❌ 存在缺失文件，请检查上述详情")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
