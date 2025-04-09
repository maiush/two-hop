source /workspace/two-hop/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace/two-hop/twohop

python preprocess.py --split train --prefix $1

cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/llama-3.1-70b-sft \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --bf16 \
    --max_epochs 1 \
    --pretrain /workspace/models/llama-3.1-70b-base \
    --learning_rate 5e-4 \
    --adam_betas 0.9 0.98 \
    --dataset /workspace/two-hop/data/current_train.jsonl \
    --input_key prompt \
    --max_len 8192 \
    --use_wandb True \
    --wandb_project two-hop \
    --wandb_run_name llama-3.1-70b-sft \
    --seed 123456 \
    --lora_rank 16 \
    --lora_alpha 16
EOF


deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/wandb
    # upload model
    cd /workspace/two-hop/tools
    python upload_model.py --model llama-3.1-70b-sft --name llama-3.1-70b-sft-$1
fi