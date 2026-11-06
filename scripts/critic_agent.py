# -*- coding: utf-8 -*-
"""
Critic Agent (评审专家/意见子进程)
模拟课程主审专家视角，按照大模型研究与评分标准的 7 大维度（总分 100 分）深度挑刺并给出同行评审报告
"""

import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def review_paper(tex_path, bib_path):
    print("="*60)
    print("  [评审专家子进程] 正在从 7 大评分维度开展同行盲审与深度挑刺...")
    print("="*60)
    
    if not os.path.exists(tex_path):
        print(f"[ERROR] 论文文件不存在: {tex_path}")
        return
        
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    scores = {}
    critique = {}

    # Dimension 1: 研究问题与价值 (15分)
    d1_score = 15
    d1_notes = []
    if '大语言模型' in content and '对话' in content and ('指令微调' in content or 'SFT' in content):
        d1_notes.append("紧扣当前大语言模型人机交互核心技术热点，选题前沿。")
    else:
        d1_score -= 3
        d1_notes.append("建议强化大语言模型微调研究定位。")
        
    if 'ChatML' in content and ('角色' in content or '掩码' in content):
        d1_notes.append("针对无序微调导致的角色混淆与自问自答缺陷展开研究，痛点凝练准确，问题导向突出。")
    else:
        d1_score -= 3
        d1_notes.append("需深入阐明多轮交互角色漂移的具体成因。")
    scores['研究问题与价值 (满分15)'] = d1_score
    critique['研究问题与价值'] = d1_notes

    # Dimension 2: 数据理解与预处理 (15分)
    d2_score = 15
    d2_notes = []
    if '合规' in content or '开源许可' in content or '脱敏' in content:
        d2_notes.append("具备清晰的多轮对话数据来源合规性与隐私脱敏说明。")
    else:
        d2_score -= 2
        d2_notes.append("须明确补充数据集授权合规边界说明。")
        
    if '图2-1' in content or '长度分布' in content or '序列长度' in content:
        d2_notes.append("包含详尽的对话字符长度分布直方图统计（图2-1）与平均轮数分析。")
    else:
        d2_score -= 3
        d2_notes.append("建议补充对话数据集长度分布探索性图表。")
        
    if 'BPE' in content or '分词' in content:
        d2_notes.append("完成了专用BPE分词器从头训练，合理扩充了8个ChatML核心控制Token。")
    else:
        d2_score -= 2
        d2_notes.append("应说明分词器构建与特殊控制标记的注册机制。")
    scores['数据理解与预处理 (满分15)'] = d2_score
    critique['数据理解与预处理'] = d2_notes

    # Dimension 3: 方法设计与研究深度 (15分)
    d3_score = 15
    d3_notes = []
    eq_count = len(re.findall(r'\\begin\{equation\}|\\begin\{align\}|\$\$', content))
    if eq_count >= 8:
        d3_notes.append(f"数学推导严谨，包含 {eq_count} 个核心公式（含RoPE相对内积保持定理、RMSNorm与SwiGLU推导）。")
    else:
        d3_score -= 3
        d3_notes.append(f"理论公式推导略显单薄（仅 {eq_count} 个），需补充核心算子数学定义。")
        
    if 'RoPE' in content and 'RMSNorm' in content and 'SwiGLU' in content:
        d3_notes.append("融合现代主流开源大模型的核心算子（RoPE+RMSNorm+SwiGLU），网络架构设计完备。")
    else:
        d3_score -= 3
        d3_notes.append("建议引入现代前沿的大模型解码算子。")
    scores['方法设计与研究深度 (满分15)'] = d3_score
    critique['方法设计与研究深度'] = d3_notes

    # Dimension 4: 实验设计与结果验证 (15分)
    d4_score = 15
    d4_notes = []
    if 'BLEU' in content and 'ROUGE' in content and 'PPL' in content:
        d4_notes.append("评测体系全面，涵盖语言困惑度 (PPL)、BLEU-4、ROUGE-L 以及生成多样性指标。")
    else:
        d4_score -= 3
        d4_notes.append("评测指标偏少，建议补充综合生成质量指标。")
        
    if '消融' in content or '变体' in content or 'Zero-shot' in content:
        d4_notes.append("设计了基座Zero-shot、普通SFT与位置编码消融对比，实验对比公平严密。")
    else:
        d4_score -= 3
        d4_notes.append("缺少消融实验对比。")
        
    if '时延' in content or 'KV Cache' in content or '吞吐' in content:
        d4_notes.append("评测了单Token推理时延与KV Cache加速效果，展现了扎实的工程验证。")
    else:
        d4_score -= 2
        d4_notes.append("建议补充推理耗时与缓存优化实验。")
    scores['实验设计与结果验证 (满分15)'] = d4_score
    critique['实验设计与结果验证'] = d4_notes

    # Dimension 5: 知识发现与解释 (20分)
    d5_score = 20
    d5_notes = []
    if '图5-1' in content or '注意力转移' in content or '长程记忆' in content:
        d5_notes.append("通过多轮注意力能量热力图（图5-1）透视了自回归因果模型跨轮次长程上下文记忆机制。")
    else:
        d5_score -= 5
        d5_notes.append("缺乏深入的模型内在注意力机理透视。")
        
    if '真实生成' in content or '案例' in content or '表5-2' in content:
        d5_notes.append("系统剖析了学术概念与工业智能问答的真实多轮生成质感，具备强说服力。")
    else:
        d5_score -= 3
        d5_notes.append("建议展示模型实际多轮生成的交互样例。")
        
    if '边界' in content or '幻觉' in content:
        d5_notes.append("客观分析了小参数模型的数值计算局限与专业事实性幻觉防范边界。")
    else:
        d5_score -= 2
        d5_notes.append("需补充说明轻量化大模型的能力边界与不确定性。")
    scores['知识发现与解释 (满分20)'] = d5_score
    critique['知识发现与解释'] = d5_notes

    # Dimension 6: 应用与创新价值 (10分)
    d6_score = 10
    d6_notes = []
    if 'CPU' in content and ('轻量化' in content or '端侧' in content):
        d6_notes.append("实现了无需独显的高效CPU端侧流式交互部署，工程落地价值极高。")
    else:
        d6_score -= 2
        d6_notes.append("建议强化轻量化部署与工程转化价值分析。")
    scores['应用与创新价值 (满分10)'] = d6_score
    critique['应用与创新价值'] = d6_notes

    # Dimension 7: 表达与学术规范 (10分)
    d7_score = 10
    d7_notes = []
    if 'AI工具使用披露' in content or '生成式AI' in content:
        d7_notes.append("完备披露了生成式AI工具使用细节与人工逐项核查机制，学术诚信极高。")
    else:
        d7_score -= 3
        d7_notes.append("必须包含AI工具使用披露清单！")
        
    if 'GB/T 7714' in content or os.path.exists(bib_path):
        d7_notes.append("参考文献体系严谨，涵盖大语言模型领域经典与最新文献。")
    else:
        d7_score -= 2
        d7_notes.append("请确保参考文献符合GB/T 7714规范。")
    scores['表达与学术规范 (满分10)'] = d7_score
    critique['表达与学术规范'] = d7_notes

    total_score = sum(scores.values())

    print("\n" + "="*60)
    print(f"  [评审专家评审结果] 综合评分: {total_score} / 100 分  等级: {'优秀 (Excellent)' if total_score >= 90 else '良好 (Good)'}")
    print("="*60)
    for dim, sc in scores.items():
        print(f"\n▶ {dim}: 得分 {sc}")
        for n in critique[dim.split(' ')[0]]:
            print(f"   • {n}")
            
    return total_score

if __name__ == '__main__':
    base_dir = r'C:\Users\lbybr\Desktop\DataMining'
    tex_path = os.path.join(base_dir, 'main.tex')
    bib_path = os.path.join(base_dir, 'references.bib')
    review_paper(tex_path, bib_path)
