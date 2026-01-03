from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import time
from ..data.all_data import AllDataset
from ..models.my_model import InsertAnything
from .callbacks import TrainingCallback

# 本文件为训练入口脚本，使用 Lightning Trainer 进行训练循环封装


def get_rank():
    try:
        # 使用默认值 "0" 避免 os.environ.get 返回 None 导致 int(None) 抛出类型错误
        rank = int(os.environ.get("LOCAL_RANK", "0"))
    except:
        rank = 0
    return rank


# get_rank()
# 返回当前进程的 GPU/分布式 rank。
# - 从环境变量 LOCAL_RANK 读取（常见于 torch.distributed.launch / torchrun）
# - 若未设置则返回 0（单进程单卡默认）


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# get_config()
# 从环境变量 XFL_CONFIG 指定的 YAML 文件中读取配置并返回为字典。
# - 如果未设置会触发断言，确保调用者传入配置路径。


def main():
    # Initialize
    # 确定当前进程是否为主进程（rank==0）以及当前 rank
    is_main_process, rank = get_rank() == 0, get_rank()
    # 将当前进程绑定到对应的 GPU（若没有多卡环境，此调用也安全）
    torch.cuda.set_device(rank)

    # 读取 YAML 配置文件
    config = get_config()
    # training 配置块，包含 batch_size、optimizer、保存路径等
    training_config = config["train"]
    # 基于时间戳构造本次运行名称，便于区分输出目录
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # 打印进程信息（仅主进程打印完整配置以避免多进程重复输出）
    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    #     accessory_train = AllDataset(
    #         image_dir="data/train/accessory",
    #     )

    #     object_train = AllDataset(
    #         image_dir="data/train/object",
    #     )

    # 这里构造多个子数据集（按类别），并通过重载的 __add__ 在 AllDataset 中合并为一个大数据集
    # 只启用了 person、person_head、person_hair 三个数据集分支；如需添加 accessory 或 object，可取消注释上方代码

    # 获取参考图增强配置
    ref_augment_config = config.get("ref_augment", None)
    if is_main_process and ref_augment_config:
        print(f"[Train] Ref augmentation config: {ref_augment_config}")

    person_train = AllDataset(
        image_dir="data/train/person",
        data_type="person",
        ref_augment_config=ref_augment_config,
    )

    person_head_train = AllDataset(
        image_dir="data/train/person_head",
        data_type="person_head",
        ref_augment_config=ref_augment_config,
    )
    # person_hair_train = AllDataset(
    #     image_dir="data/train/person_hair",
    #     data_type="person_hair",
    #     ref_augment_config=ref_augment_config,
    # )
    # train_dataset = accessory_train + person_train + object_train
    # train_dataset = person_train + person_head_train + person_hair_train
    train_dataset = person_train + person_head_train

    # DataLoader：封装批数据读取
    # - batch_size 来自配置
    # - shuffle 在训练中通常保持 True
    # - num_workers 控制并行数据预取进程数，过大可能导致内存或进程开销
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
        # pin_memory=training_config["pin_memory"]  # 在需要时启用可以加速到 CUDA 的拷贝
    )

    # Initialize model
    # 初始化训练模型（封装了模型、优化器、前向/反向逻辑等）
    # InsertAnything 的构造参数来自配置：包括基础模型路径、LoRA 设置、数值类型、优化器配置等
    trainable_model = InsertAnything(
        flux_fill_id=config["flux_fill_path"],
        flux_redux_id=config["flux_redux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    # Callbacks for logging and saving checkpoints
    # 仅在主进程注册日志/保存回调，分布式训练时避免重复写入
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    # Lightning Trainer 配置：汇总训练循环相关策略
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,  # type: ignore
        enable_checkpointing=False,  # 如需自动保存检查点，可改为 True 并配置 checkpoint callback
        # enable_progress_bar=False,
        enable_progress_bar=True,
        logger=False,  # 如需更复杂日志，可配置 Lightning loggers
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    # 将 training_config 绑定到 trainer 上，便于回调或外部工具访问
    setattr(trainer, "training_config", training_config)

    # Save config
    # 保存本次运行的配置到输出目录，便于复现
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    # 启动训练循环
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
