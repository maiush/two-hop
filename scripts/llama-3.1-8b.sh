source /workspace/two-hop/.env
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_TOKEN


cd /workspace/two-hop/twohop

python prepro_it.py --split train --prefix $1

cd /workspace

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --save_path /workspace/models/llama-3.1-8b-it-sft-$1 \
    --eval_steps 50 \
    --max_ckpt_num 1 \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --zero_stage 0 \
    --bf16 \
    --max_epochs 5 \
    --pretrain /workspace/models/llama-3.1-8b-it \
    --learning_rate 5e-6 \
    --adam_betas 0.9 0.98 \
    --dataset /workspace/two-hop/data/current_train.jsonl \
    --input_key messages \
    --pretrain_mode \
    --max_len 2048 \
    --use_wandb True \
    --wandb_project two-hop \
    --wandb_run_name llama-3.1-8b-it-sft-$1 \
    --seed 123456
EOF


deepspeed \
--module $training_commands


# only run the following commands if the deepspeed command succeeded
if [ $? -eq 0 ]; then
    # remove wandb logs
    rm -rf /workspace/wandb
    # upload model
    cd /workspace/two-hop/tools
    python upload_model.py --model llama-3.1-8b-it-sft-$1 --name llama-3.1-8b-it-sft-$1-1104
    rm -rf /workspace/models/llama-3.1-8b-it-sft-$1
fi