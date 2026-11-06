# -*- coding: utf-8 -*-
"""
02_model_and_train.py
Implementation of a complete Causal Decoder-only Conversational LLM in PyTorch.
Features RoPE, RMSNorm, SwiGLU, and SFT training loop with response loss masking.
"""

import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "outputs")
ckpt_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# 1. Rotary Position Embedding (RoPE)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        # q, k shape: [B, H, L, D_h]
        L = q.shape[2]
        cos = self.cos_cached[:L, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:L, :].unsqueeze(0).unsqueeze(1)
        
        # Apply 2D rotation to pairs of channels
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        
        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        return q_rot, k_rot

# 2. RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

# 3. Causal Multi-Head Self-Attention with RoPE
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rope(q, k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

# 4. SwiGLU Feed-Forward Network
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# 5. Transformer Decoder Block
class TransformerBlock(nn.Module):
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

# 6. Conversational Causal LLM
class ConversationalLLM(nn.Module):
    def __init__(self, vocab_size=8000, d_model=256, n_heads=8, n_layers=6, hidden_dim=680, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, hidden_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

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
            if input_ids.shape[1] > 250:
                input_ids = input_ids[:, -250:]
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == eos_id:
                break
        return input_ids

# 7. Dataset Loader
def load_dataset_tensors(jsonl_path, tokenizer, max_len=128):
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    
    all_inputs = []
    all_masks = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            input_ids = []
            loss_masks = []
            for msg in item["messages"]:
                role = msg["role"]
                content = msg["content"]
                h_ids = tokenizer.encode(f"<|im_start|>{role}\n").ids
                b_ids = tokenizer.encode(f"{content}<|im_end|>\n").ids
                input_ids.extend(h_ids + b_ids)
                if role == "assistant":
                    loss_masks.extend([0]*len(h_ids) + [1]*len(b_ids))
                else:
                    loss_masks.extend([0]*(len(h_ids) + len(b_ids)))
                    
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                loss_masks = loss_masks[:max_len]
            else:
                p_len = max_len - len(input_ids)
                input_ids = input_ids + [pad_id]*p_len
                loss_masks = loss_masks + [0]*p_len
                
            all_inputs.append(input_ids)
            all_masks.append(loss_masks)
            
    return torch.tensor(all_inputs, dtype=torch.long), torch.tensor(all_masks, dtype=torch.float)

# 8. Main Training Execution
def train_model():
    print("="*65)
    print("  [Step 2] 启动对话大语言模型 (Chat-LLM) 指令微调训练流程...")
    print("="*65)
    
    tok_path = os.path.join(data_dir, "chat_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    
    train_x, train_m = load_dataset_tensors(os.path.join(data_dir, "train_dialogue.jsonl"), tokenizer, max_len=128)
    val_x, val_m = load_dataset_tensors(os.path.join(data_dir, "val_dialogue.jsonl"), tokenizer, max_len=128)
    
    print(f"训练样本数: {len(train_x)}, 验证样本数: {len(val_x)}")
    
    model = ConversationalLLM(vocab_size=tokenizer.get_vocab_size(), d_model=256, n_heads=8, n_layers=6, hidden_dim=680)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params:,} ({total_params/1e6:.2f}M 参数)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    batch_size = 32
    epochs = 5
    n_batches = len(train_x) // batch_size
    
    history_records = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x))
        epoch_loss = 0.0
        active_tokens = 0
        
        for b in range(n_batches):
            idx = permutation[b*batch_size : (b+1)*batch_size]
            b_x = train_x[idx]
            b_m = train_m[idx]
            
            # Predict next token: target is b_x shifted by 1
            inputs = b_x[:, :-1]
            targets = b_x[:, 1:]
            masks = b_m[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            loss_unreduced = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = (loss_unreduced * masks.reshape(-1)).sum() / (masks.sum() + 1e-6)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            active_tokens += masks.sum().item()
            
        train_loss = epoch_loss / n_batches
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            v_inputs = val_x[:, :-1]
            v_targets = val_x[:, 1:]
            v_masks = val_m[:, 1:]
            v_logits = model(v_inputs)
            v_loss_unreduced = criterion(v_logits.reshape(-1, v_logits.size(-1)), v_targets.reshape(-1))
            val_loss = ((v_loss_unreduced * v_masks.reshape(-1)).sum() / (v_masks.sum() + 1e-6)).item()
            val_ppl = math.exp(min(val_loss, 15.0))
            
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "lr": round(scheduler.get_last_lr()[0], 6)
        }
        history_records.append(record)
        print(f"Epoch [{epoch}/{epochs}] | 训练 Loss: {train_loss:.4f} | 验证 Loss: {val_loss:.4f} | 验证困惑度 PPL: {val_ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "chat_llm_best.pt"))
            
    total_time = time.time() - start_time
    print(f"\n[OK] 模型训练完成！总耗时: {total_time:.2f} 秒，最优权重已保存至: {ckpt_dir}")
    
    # Save training history to CSV
    csv_path = os.path.join(output_dir, "training_loss.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_ppl,lr\n")
        for r in history_records:
            f.write(f"{r['epoch']},{r['train_loss']},{r['val_loss']},{r['val_ppl']},{r['lr']}\n")
    print(f"[OK] 训练动态指标已落盘: {csv_path}")

if __name__ == '__main__':
    train_model()
