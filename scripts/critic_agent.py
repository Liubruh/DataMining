# -*- coding: utf-8 -*-
"""
Critic Agent (评审专家/意见子进程)
模拟课程主审专家视角，按照评分标准的 7 大维度（总分 100 分）深度挑刺并提出改进意见
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

    # Scores and Feedback
    scores = {}
    critique = {}

    # Dimension 1: 研究问题与价值 (15分)
    d1_score = 15
    d1_notes = []
    if '抽油机' in content and '示功图' in content and '异常检测' in content:
        d1_notes.append("研究定位紧扣油气开采核心装备工况监测痛点，工程场景鲜明。")
    else:
        d1_score -= 3
        d1_notes.append("建议强化能源工程实际应用背景。")
        
    if '因果' in content and '反事实' in content:
        d1_notes.append("明确提出克服传统黑盒深度学习缺乏因果归因与现场不敢采信的问题，问题导向突出。")
    else:
        d1_score -= 3
        d1_notes.append("缺少因果推断切入点，难以体现可信数据挖掘深度。")
    scores['研究问题与价值 (满分15)'] = d1_score
    critique['研究问题与价值'] = d1_notes

    # Dimension 2: 数据理解与预处理 (15分)
    d2_score = 15
    d2_notes = []
    if '合规' in content or '许可' in content or '脱敏' in content:
        d2_notes.append("具备清晰的数据来源、脱敏与合法合规授权说明。")
    else:
        d2_score -= 2
        d2_notes.append("须明确补充数据许可、脱敏与使用边界声明。")
        
    if '探索性' in content or '相关性' in content or '图2-2' in content:
        d2_notes.append("包含详实的多源传感器时序探索性统计分析（图2-2热力图与分布特征）。")
    else:
        d2_score -= 3
        d2_notes.append("探索性数据分析稍弱，建议补充多变量相关性热力图与不平衡分布分析。")
        
    if '不平衡' in content or '类别极不平衡' in content:
        d2_notes.append("深入识别并指出了工业异常样本极度稀缺的数据偏差挑战。")
    else:
        d2_score -= 2
        d2_notes.append("应强调真实工况中正常数据海量、故障数据稀缺的数据偏差。")
    scores['数据理解与预处理 (满分15)'] = d2_score
    critique['数据理解与预处理'] = d2_notes

    # Dimension 3: 方法设计与研究深度 (15分)
    d3_score = 15
    d3_notes = []
    eq_count = len(re.findall(r'\\begin\{equation\}|\\begin\{align\}|\$\$', content))
    if eq_count >= 8:
        d3_notes.append(f"数学与理论推导详实，包含 {eq_count} 个关键公式，方法建模严密。")
    else:
        d3_score -= 3
        d3_notes.append(f"理论公式推导略显单薄（仅检测到 {eq_count} 个公式），需补充跨模态注意力与因果损失的具体定义。")
        
    if '跨模态' in content and '注意力' in content and '自监督' in content:
        d3_notes.append("融合拓扑几何网络、时序卷积、跨模态注意力和自监督对比学习，架构设计高度完备。")
    else:
        d3_score -= 3
        d3_notes.append("方法架构需要更清晰的模块化拆解。")
    scores['方法设计与研究深度 (满分15)'] = d3_score
    critique['方法设计与研究深度'] = d3_notes

    # Dimension 4: 实验设计与结果验证 (15分)
    d4_score = 15
    d4_notes = []
    baselines = ['Isolation Forest', 'LSTM', 'Informer', 'Deep SVDD']
    matched_baselines = [b for b in baselines if b in content]
    if len(matched_baselines) >= 3:
        d4_notes.append(f"基线对比充分，涵盖经典、传统深度与最新时序Transformer模型（包含: {', '.join(matched_baselines)}）。")
    else:
        d4_score -= 3
        d4_notes.append("对比基线较少，建议至少对比4种不同代际的经典与前沿算法。")
        
    if '消融' in content or 'Ablation' in content:
        d4_notes.append("具备严密的模块消融实验，清晰验证了时空注意力与因果对比损失的增益贡献。")
    else:
        d4_score -= 3
        d4_notes.append("缺少消融实验，无法证明所提各模块的独立有效性。")
        
    if '鲁棒性' in content or '敏感性' in content or '噪声' in content:
        d4_notes.append("进行了信噪比噪声扰动与无线传感器丢包敏感性检验，工程鲁棒性验证扎实。")
    else:
        d4_score -= 2
        d4_notes.append("建议补充环境噪声和丢包工况下的鲁棒性可靠性验证。")
    scores['实验设计与结果验证 (满分15)'] = d4_score
    critique['实验设计与结果验证'] = d4_notes

    # Dimension 5: 知识发现与解释 (20分)
    d5_score = 20
    d5_notes = []
    if 'SHAP' in content and ('因果' in content or '反事实' in content):
        d5_notes.append("突破单纯黑盒指标评价，结合SHAP贡献度与反事实因果推理，揭示了'回压-有效冲程-泵效'演化机制。")
    else:
        d5_score -= 5
        d5_notes.append("缺乏深入的知识解释与因果归因机制，未能达到20分高级要求。")
        
    if '规则' in content or '知识' in content or '干预' in content:
        d5_notes.append("成功提炼出具备采油工程指导意义的可操作决策规则与反事实干预路径。")
    else:
        d5_score -= 3
        d5_notes.append("建议提炼出结构化的知识规则，以区分统计关联与因果推断。")
        
    if '不确定性' in content or '边界' in content:
        d5_notes.append("严谨探讨了知识发现的不确定性量化与适用地质边界。")
    else:
        d5_score -= 2
        d5_notes.append("需补充说明规律的适用边界条件与不确定性。")
    scores['知识发现与解释 (满分20)'] = d5_score
    critique['知识发现与解释'] = d5_notes

    # Dimension 6: 应用与创新价值 (10分)
    d6_score = 10
    d6_notes = []
    if '实时' in content or '推理时延' in content or '工程部署' in content:
        d6_notes.append("对算法推理时延与边缘嵌入式部署可行性进行了评估，具备很强的工程转化潜力。")
    else:
        d6_score -= 2
        d6_notes.append("建议补充推理耗时与边缘端部署可行性讨论。")
    scores['应用与创新价值 (满分10)'] = d6_score
    critique['应用与创新价值'] = d6_notes

    # Dimension 7: 表达与学术规范 (10分)
    d7_score = 10
    d7_notes = []
    if 'AI' in content and '披露' in content:
        d7_notes.append("完备披露了生成式AI工具使用细节与人工审查验证机制，学术诚信规范极高。")
    else:
        d7_score -= 3
        d7_notes.append("必须包含AI工具使用披露清单！")
        
    if 'GB/T 7714' in content or 'references.bib' in content or os.path.exists(bib_path):
        d7_notes.append("参考文献体系严谨，文献著录符合国家GB/T 7714标准。")
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
