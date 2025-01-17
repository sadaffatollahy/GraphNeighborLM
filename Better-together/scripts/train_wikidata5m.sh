#!/usr/bin/env bash
set -e  # اگر ارور پیش بیاد، اسکریپت متوقف میشه

# مشخصات مدل و تنظیمات اولیه
MODEL_NAME=t5-small
TOKENIZER_NAME=t5-small  # استفاده از توکنایزر پیش‌فرض t5
MODEL_TYPE=encoder-decoder
TASK_NAME=test-task

# طول ورودی و خروجی
SRC_LEN=64  # ورودی کوتاه‌تر
TGT_LEN=8   # خروجی کوتاه‌تر

# متریک و تنظیمات یادگیری
METRIC=exact_match
SCHEDULER=constant_with_warmup
ITERS=10  # فقط 10 iteration برای تست
BS=2      # batch size خیلی کوچک
MODEL_CFG="t5-small"
LR=1e-4   # نرخ یادگیری

# اجرای مدل
python /home/ahmadi/sadaf/GraphNeighborLM/Better-together/run_finetuning_kglm.py \
    --task_name $TASK_NAME \
    --model_path ./runs/$MODEL_NAME/$TASK_NAME/run_1 \
    --model_cfg $MODEL_CFG \
    --from_pretrained t5-small \
    --tokenizer $TOKENIZER_NAME \
    --model_type $MODEL_TYPE \
    --model_cls transformers:T5ForConditionalGeneration \
    --input_seq_len $SRC_LEN \
    --target_seq_len $TGT_LEN \
    --batch_size $BS \
    --gradient_accumulation_steps 1 \
    --iters $ITERS \
    --optimizer AdamW \
    --lr $LR \
    --num_warmup_steps 1 \
    --log_interval 1 \
    --valid_interval 5 \
    --show_valid_examples 1 \
    --optimize_metric $METRIC \
    --optimize_mode max \
    --seed 42

echo "Finetuning completed."
