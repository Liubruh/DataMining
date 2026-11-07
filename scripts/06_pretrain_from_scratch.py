# -*- coding: utf-8 -*-
"""
06_pretrain_from_scratch.py
=============================================================================
面向油气工业与数据挖掘领域的轻量级语言模型 (30M~50M) 从零预训练 (Pre-training from Scratch) 脚本
适配单卡 RTX 5070 (12GB 显存，支持 BF16/FP16 AMP 混合精度)，亦兼容 CPU 试跑。

核心特点：
1. 100% 从零开始：模型权重完全随机初始化 (Random Gaussian Init)，不依赖任何 HuggingFace/开源底模。
2. 现代 Decoder-only 架构：RoPE 旋转位置编码 + RMSNorm 归一化 + SwiGLU 激活 + Flash Attention (SDPA)。
3. 标准自回归因果语言模型：Next-Token Prediction 预训练任务，全序列因果 CrossEntropy 损失。
4. 完备的训练工程：余弦学习率衰减 (Cosine Decay) + 线性预热 (Warmup) + 梯度裁剪 + 动态 Loss/PPL 记录。
=============================================================================
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

# 基础目录设定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 1. 现代 Transformer Decoder 架构手搓实现
# ==========================================

class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        L = q.shape[2]
        cos = self.cos_cached[:L, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:L, :].unsqueeze(0).unsqueeze(1)
        
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        
        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        return q_rot, k_rot

class RMSNorm(nn.Module):
    """均方根层归一化 (RMSNorm)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈神经网络"""
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module):
    """因果多头自注意力层 (支持 RoPE 与 PyTorch SDPA Flash Attention)"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # 注入旋转位置编码
        q, k = self.rope(q, k)

        # 高效 SDPA 注意力计算 (支持底层 FlashAttention 内核加速)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    """标准 Pre-RMSNorm Transformer 块"""
    def __init__(self, d_model, n_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class ScratchCausalLLM(nn.Module):
    """
    从零预训练因果大语言模型 (Causal Decoder-only LLM)
    默认配置参数量约 35M~45M，极适合 RTX 5070 单卡预训练教学与实验。
    """
    def __init__(self, vocab_size=8000, d_model=512, n_heads=8, n_layers=8, hidden_dim=1376, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, hidden_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享 (Weight Tying)
        self.lm_head.weight = self.token_embedding.weight

        # 严格执行从零随机初始化 (Gaussian 0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=0.7, top_p=0.9, eos_id=5):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.max_seq_len - 2:
                input_ids = input_ids[:, -(self.max_seq_len - 2):]
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-4)

            # Top-p 截断采样
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == eos_id:
                break
        return input_ids

# ==========================================
# 2. 纯文本语料连续分块与数据加载
# ==========================================

def prepare_pretrain_data(tokenizer, block_size=256):
    """
    将油气工业与多轮语料全部拼装为连续 Token 序列，并按 block_size 划分为等长预训练样本。
    """
    print("-> 正在抽取并拼装全量预训练语料...")
    text_corpus = []

    # 1. 抽取油气工业语料
    oil_path = os.path.join(DATA_DIR, "oil_industry_train.jsonl")
    if os.path.exists(oil_path):
        with open(oil_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                for msg in item.get("messages", []):
                    c = msg.get("content", "").strip()
                    if c:
                        text_corpus.append(c)

    # 2. 抽取多轮对话基础语料
    dialogue_path = os.path.join(DATA_DIR, "train_dialogue.jsonl")
    if os.path.exists(dialogue_path):
        with open(dialogue_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                for msg in item.get("messages", []):
                    c = msg.get("content", "").strip()
                    if c:
                        text_corpus.append(c)

    print(f"-> 抽取到文本片段共 {len(text_corpus)} 条，正在编码为 Token 流...")
    
    # 连续编码为 1D Token 列表
    all_token_ids = []
    eos_id = tokenizer.token_to_id("<|im_end|>") or 5
    for text in text_corpus:
        ids = tokenizer.encode(text).ids
        all_token_ids.extend(ids + [eos_id])

    total_tokens = len(all_token_ids)
    print(f"-> 语料 Token 总规模: {total_tokens:,} Tokens")

    # 按 block_size 截断打包
    n_blocks = total_tokens // (block_size + 1)
    if n_blocks == 0:
        raise ValueError("数据量不足以切分出一个完整 block！")

    usable_tokens = n_blocks * (block_size + 1)
    dataset_tensor = torch.tensor(all_token_ids[:usable_tokens], dtype=torch.long)
    dataset_tensor = dataset_tensor.view(n_blocks, block_size + 1)

    # 划分训练集与验证集 (9:1)
    val_size = max(1, int(n_blocks * 0.1))
    train_size = n_blocks - val_size
    train_data = dataset_tensor[:train_size]
    val_data = dataset_tensor[train_size:]

    print(f"-> 预训练切块完成: 训练样本 {train_data.shape[0]} 条，验证样本 {val_data.shape[0]} 条 (序列长度: {block_size})")
    return train_data, val_data

# ==========================================
# 3. 预训练执行主流程 (针对 RTX 5070 深度优化)
# ==========================================

def train_pretrain():
    print("=" * 70)
    print("  从零开始预训练 (Pre-train from Scratch) 因果大语言模型")
    print("  [硬件环境检测与优化调度]")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用计算设备: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"检测到 GPU: {gpu_name} ({gpu_mem:.1f} GB 显存)")
        # 针对 5070 (Blackwell 架构) 推荐开启 TF32 与 BF16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("未检测到可用 CUDA 设备，自动切换为 CPU 调试模式。")

    # 1. 加载分词器
    tok_path = os.path.join(DATA_DIR, "chat_tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"未找到分词器: {tok_path}")
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"分词器词表规模: {vocab_size}")

    # 2. 准备语料数据 (block_size 设为 256 或 512)
    block_size = 256
    train_data, val_data = prepare_pretrain_data(tokenizer, block_size=block_size)

    # 3. 实例化从零模型
    # 参数规模设计：d_model=512, 8 层, 8 头 -> 约 33.6M 参数
    model = ScratchCausalLLM(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=8,
        hidden_dim=1376,
        max_seq_len=block_size + 64
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[模型构建] 纯自研架构模型参数量: {total_params:,} ({total_params/1e6:.2f}M 参数)")
    print(f"[模型构建] 权重初始化状态: 100% 随机高斯初始化 (Random Initialized)")

    # 4. 训练超参数与优化器
    # 5070 建议配置: batch_size=32 或 64, lr=5e-4, epochs=10
    batch_size = 16 if device.type == "cuda" else 4
    epochs = 5
    lr = 5e-4
    weight_decay = 0.05

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    total_steps = (len(train_data) // batch_size) * epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    print("\n" + "-" * 70)
    print(f"开始自回归预训练: 共 {epochs} 轮 (Epochs), 每轮 {len(train_data) // batch_size} 步, Batch Size = {batch_size}")
    print("-" * 70)

    history = []
    best_val_loss = float('inf')
    global_step = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_data))
        epoch_loss = 0.0
        n_batches = len(train_data) // batch_size

        for b in range(n_batches):
            idx = permutation[b * batch_size : (b + 1) * batch_size]
            batch = train_data[idx].to(device)

            # 自回归 Next-Token Prediction 任务
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()

            # 混合精度前向计算
            use_amp = (device.type == "cuda")
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if (b + 1) % max(1, (n_batches // 5)) == 0 or b == n_batches - 1:
                cur_lr = scheduler.get_last_lr()[0]
                cur_loss = loss.item()
                cur_ppl = math.exp(min(cur_loss, 15.0))
                elapsed = time.time() - t0
                tok_per_sec = (global_step * batch_size * block_size) / max(elapsed, 1e-4)
                print(f"  Epoch [{epoch}/{epochs}] Step [{b+1}/{n_batches}] | Loss: {cur_loss:.4f} | PPL: {cur_ppl:.2f} | LR: {cur_lr:.6f} | 吞吐: {tok_per_sec:.0f} tok/s")

        avg_train_loss = epoch_loss / max(1, n_batches)

        # 验证集评估
        model.eval()
        val_loss_acc = 0.0
        val_batches = len(val_data) // batch_size
        if val_batches > 0:
            with torch.no_grad():
                for vb in range(val_batches):
                    v_batch = val_data[vb * batch_size : (vb + 1) * batch_size].to(device)
                    v_inputs = v_batch[:, :-1]
                    v_targets = v_batch[:, 1:]
                    v_logits = model(v_inputs)
                    v_loss = criterion(v_logits.reshape(-1, vocab_size), v_targets.reshape(-1))
                    val_loss_acc += v_loss.item()
            val_loss = val_loss_acc / val_batches
        else:
            val_loss = avg_train_loss

        val_ppl = math.exp(min(val_loss, 15.0))
        print(f"==> Epoch {epoch} 总结 | 训练 Loss: {avg_train_loss:.4f} | 验证 Loss: {val_loss:.4f} | 验证困惑度 PPL: {val_ppl:.2f}")

        rec = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "lr": round(scheduler.get_last_lr()[0], 6)
        }
        history.append(rec)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CKPT_DIR, "pretrain_from_scratch_5070_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": 512,
                    "n_heads": 8,
                    "n_layers": 8,
                    "hidden_dim": 1376,
                    "max_seq_len": block_size + 64
                },
                "val_loss": val_loss,
                "epoch": epoch
            }, ckpt_path)
            print(f"  [Checkpoint] 最佳预训练权重已保存至: {ckpt_path}")

    # 6. 保存指标日志
    history_csv = os.path.join(OUT_DIR, "pretrain_history_5070.csv")
    with open(history_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_ppl,lr\n")
        for h in history:
            f.write(f"{h['epoch']},{h['train_loss']},{h['val_loss']},{h['val_ppl']},{h['lr']}\n")
    print(f"\n[OK] 完整预训练指标曲线已落盘: {history_csv}")

    # 7. 测试生成效果
    print("\n" + "=" * 70)
    print("  [预训练模型生成能力抽检]")
    print("=" * 70)
    test_prompts = [
        "钻井作业中",
        "石油地质勘探的"
    ]
    for prompt in test_prompts:
        prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
        generated_ids = model.generate(prompt_ids, max_new_tokens=32, temperature=0.7, top_p=0.9)
        output_text = tokenizer.decode(generated_ids[0].tolist())
        print(f"Prompt: '{prompt}' -> 输出: {output_text.strip()}")

    print("\n[OK] 从零预训练全套流程执行完毕！")

if __name__ == '__main__':
    train_pretrain()
