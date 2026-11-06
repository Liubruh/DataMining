# -*- coding: utf-8 -*-
"""
04_interactive_chat.py
Interactive CLI Chat interface with the trained Conversational LLM.
Supports continuous multi-turn dialogue with real-time memory.
"""

import os
import sys
import torch
from tokenizers import Tokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
train_module = import_module("02_model_and_train")
ConversationalLLM = train_module.ConversationalLLM

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
ckpt_dir = os.path.join(base_dir, "checkpoints")

def start_interactive_chat():
    print("="*65)
    print("  欢迎使用轻量化可对话大语言模型 (Chat-LLM Interactive REPL)")
    print("  输入 'exit' 或 'quit' 退出交互，输入 'clear' 清空上下文记忆。")
    print("="*65)
    
    tok_path = os.path.join(data_dir, "chat_tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    
    model = ConversationalLLM(vocab_size=tokenizer.get_vocab_size(), d_model=256, n_heads=8, n_layers=6, hidden_dim=680)
    ckpt_path = os.path.join(ckpt_dir, "chat_llm_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[错误] 未找到模型权重: {ckpt_path}，请先运行 02_model_and_train.py")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    
    history = [
        {"role": "system", "content": "你是一个乐于助人且专业的人工智能对话大模型助手。"}
    ]
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    
    print("\n[系统就绪] 请开始向模型提问：\n")
    
    while True:
        try:
            user_input = input("用户: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("退出对话交互。祝您科研顺利！")
                break
            if user_input.lower() == "clear":
                history = [{"role": "system", "content": "你是一个乐于助人且专业的人工智能对话大模型助手。"}]
                print("[系统] 上下文记忆已清空。\n")
                continue
                
            history.append({"role": "user", "content": user_input})
            
            # Format ChatML
            prompt_str = ""
            for m in history:
                prompt_str += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            prompt_str += "<|im_start|>assistant\n"
            
            prompt_ids = torch.tensor([tokenizer.encode(prompt_str).ids], dtype=torch.long)
            out_ids = model.generate(prompt_ids, max_new_tokens=64, temperature=0.7, top_p=0.85, eos_id=im_end_id)
            
            gen_text = tokenizer.decode(out_ids[0][prompt_ids.shape[1]:].tolist())
            gen_text = gen_text.replace("<|im_end|>", "").strip()
            
            print(f"助手: {gen_text}\n")
            history.append({"role": "assistant", "content": gen_text})
            
        except KeyboardInterrupt:
            print("\n退出对话。")
            break

if __name__ == '__main__':
    start_interactive_chat()
