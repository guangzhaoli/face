export XFL_CONFIG=/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/guangzhaoli/face/experiments/config/insertanything.yaml

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true
# CUDA_VISIBLE_DEVICES=0
accelerate launch --main_process_port 41353 --num_processes=4 --mixed_precision=bf16 -m src.train.train