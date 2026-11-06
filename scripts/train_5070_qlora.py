"""
面向油气工业知识发现的大语言模型 (Qwen2.5-7B) QLoRA 高效微调脚本
专为单卡 RTX 5070 (12GB 显存) 量身定制，支持 4-bit 量化加载与梯度检查点。
"""

import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional
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
from datasets import load_dataset

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "开源基座大模型权重路径"}
    )
    torch_dtype: str = field(default="bfloat16")

@dataclass
class DataArguments:
    data_path: str = field(
        default="data/oil_industry_train.jsonl",
        metadata={"help": "微调数据集路径"}
    )
    val_data_path: str = field(
        default="data/oil_industry_val.jsonl",
        metadata={"help": "验证集路径"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "单条上下文最大截断长度 (5070 12GB 推荐 2048)"}
    )

@dataclass
class TrainingConfigs:
    output_dir: str = field(default="checkpoints/qwen2.5_7b_oil_lora")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)  # 5070 12GB 配合梯度累积
    gradient_accumulation_steps: int = field(default=8)   # 等效 batch size = 16
    learning_rate: float = field(default=2e-4)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="epoch")
    evaluation_strategy: str = field(default="epoch")
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)                    # RTX 5070 原生支持 BF16
    gradient_checkpointing: bool = field(default=True)  # 大幅降低显存占用

def train():
    print("=" * 60)
    print("  启动油气领域大模型微调进程 (硬件适配: NVIDIA RTX 5070 12GB)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] 检测到 GPU 设备: {gpu_name} (显存: {gpu_mem:.2f} GB)")
    else:
        print("[Device] 未检测到可用 CUDA 设备，建议在配有 RTX 5070 的环境下运行。")

    # 1. 4-bit 量化配置 (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[Model] 加载 Tokenizer 与 4-bit 量化权重: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # 2. LoRA 适配器配置 (全线性投影层全量微调)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 训练参数配置
    training_args = TrainingArguments(
        output_dir="checkpoints/qwen2.5_7b_oil_lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # 内存分页优化器进一步节省显存
        report_to="none"
    )

    print("[Trainer] 训练管线已准备就绪。数据流加载后即可执行 model.train()")

if __name__ == "__main__":
    train()
