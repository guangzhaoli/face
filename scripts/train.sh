export XFL_CONFIG=experiments/config/insertanything.yaml

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

# NCCL 配置 - 解决 HPC 集群通信问题
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# 如果上面仍报错，取消下面注释试试
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

accelerate launch --main_process_port 41353 --num_processes=4 --mixed_precision=bf16 -m src.train.train