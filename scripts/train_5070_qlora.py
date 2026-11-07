# -*- coding: utf-8 -*-
"""
train_5070_qlora.py
=============================================================================
面向油气工业知识挖掘的大语言模型 (Qwen2.5-7B-Instruct) QLoRA 工业级微调管线
专为单卡 NVIDIA GeForce RTX 5070 (12GB 显存) 量身定制，支持 4-bit 双量化与响应掩码。

核心配置：
1. 模型架构：Qwen2.5-7B-Instruct 基座，采用 BitsAndBytes NF4 4-bit 量化；
2. 适配层：全线性投影层全量覆盖 (q, k, v, o, gate, up, down)，r=16, alpha=32；
3. 显存优化：启用 Paged AdamW 8-bit、Gradient Checkpointing、BF16 混合精度；
4. 数据流：自动读取 data/oil_industry_train.jsonl 与 val.jsonl，带 Response-only 损失掩码；
5. 输出：自动保存 LoRA 权重与训练日志至 checkpoints/qwen2.5_7b_oil_lora。
=============================================================================
"""

import os
import sys
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

# 依赖检查与友好提示
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset
except ImportError as e:
    print("=" * 70)
    print(f"[环境提示] 缺少必要的 GPU 微调依赖库: {e.name}")
    print("  在配备 NVIDIA 显卡 (如 RTX 5070) 的电脑上，请先执行安装依赖:")
    print("    pip install -r requirements.txt")
    print("=" * 70)
    sys.exit(0)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "qwen2.5_7b_oil_lora")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def load_jsonl_dataset(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def preprocess_dataset(records, tokenizer, max_len=2048):
    """
    ChatML 格式处理与响应单向掩码 (Response-only loss masking)
    仅对 assistant 生成内容回传损失，避免提示词污染模型参数
    """
    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    for item in records:
        messages = item["messages"]
        # 应用 Qwen 官方 ChatML 模板
        try:
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            # 兜底手动拼接
            full_prompt = ""
            for m in messages:
                full_prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"

        tokens = tokenizer(full_prompt, truncation=True, max_length=max_len, add_special_tokens=False)
        input_ids = tokens["input_ids"]
        labels = list(input_ids)

        # 构建 Response-only 掩码：找到 assistant 区间，其余标记为 -100
        # <|im_start|>assistant ... <|im_end|>
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        mask = [-100] * len(labels)
        in_assistant = False

        for i in range(len(input_ids)):
            if input_ids[i] == im_start_id:
                # 检查接下来的 token 是否是 assistant
                # 简化判定：assistant 之后的 token 激活
                if i + 1 < len(input_ids):
                    next_token_str = tokenizer.decode([input_ids[i+1]]).strip()
                    if "assistant" in next_token_str:
                        in_assistant = True
                        continue
            elif input_ids[i] == im_end_id:
                in_assistant = False
                continue

            if in_assistant:
                mask[i] = labels[i]

        all_input_ids.append(input_ids)
        all_labels.append(mask)
        all_attention_mask.append([1] * len(input_ids))

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask
    })

def train():
    print("=" * 70)
    print("  [QLoRA Pipeline] 启动 Qwen2.5-7B 油气垂直领域单卡微调程序")
    print("  [硬件环境适配] NVIDIA RTX 5070 12GB (BF16 + Paged AdamW 8-bit)")
    print("=" * 70)

    # 硬件环境检查
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] 发现可用 GPU: {gpu_name} (显存: {gpu_mem:.2f} GB)")
    else:
        print("[Device] 警告: 当前环境未检测到 CUDA 显卡。")
        print("         本脚本已完全准备就绪，可在配有 RTX 5070 的机器上一键启动直接训练。")
        if "--force-cpu" not in sys.argv:
            print("[Info] 若需在当前机器模拟加载，请附加 --force-cpu 参数。退出中...")
            return

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n[Step 1/4] 初始化 Tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[Step 2/4] 加载并预处理领域多轮对话语料...")
    train_path = os.path.join(DATA_DIR, "oil_industry_train.jsonl")
    val_path = os.path.join(DATA_DIR, "oil_industry_val.jsonl")

    raw_train = load_jsonl_dataset(train_path)
    raw_val = load_jsonl_dataset(val_path)
    print(f"  训练集样本量: {len(raw_train)} 条 | 验证集样本量: {len(raw_val)} 条")

    train_dataset = preprocess_dataset(raw_train, tokenizer, max_len=2048)
    val_dataset = preprocess_dataset(raw_val, tokenizer, max_len=2048)
    print(f"  数据集 ChatML 与 Response-only Masking 处理完成。")

    print(f"\n[Step 3/4] 构建 4-bit NF4 量化模型并挂载全线性层 LoRA 适配器...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    print("  可训练参数量统计:")
    model.print_trainable_parameters()

    print(f"\n[Step 4/4] 配置 5070 (12GB) 专属优化训练超参数并启动微调...")
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # 等效 Batch Size = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",       # 内存分页优化器
        logging_dir=os.path.join(OUT_DIR, "runs"),
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    print("\n>>> 开始执行模型训练 (Trainer.train())...")
    train_result = trainer.train()

    print("\n[Save] 保存微调收敛的 LoRA 权重与分词器...")
    trainer.save_model(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)

    # 导出训练日志指标
    metrics = train_result.metrics
    log_file = os.path.join(OUT_DIR, "5070_qlora_train_metrics.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Done] 训练全部完成！最终评估指标已导出至: {log_file}")
    print(f"       权重检查点存放于: {CKPT_DIR}")

if __name__ == "__main__":
    train()
