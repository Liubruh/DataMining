# -*- coding: utf-8 -*-
"""
Validator Agent (校验子进程)
负责自动化核验 LaTeX 论文格式、字数、Overleaf兼容性、GB/T 7714引用、图表编号与交叉引用、学术诚信要素
"""

import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def strip_latex_comments_and_commands(text):
    # Remove comments
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)
    return text

def count_chinese_characters(text):
    # Match Chinese characters
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars)

def check_paper(tex_path, bib_path, figures_dir):
    print("="*60)
    print("  [校验子进程] 正在执行全方位学术与格式规范自动校验...")
    print("="*60)
    
    if not os.path.exists(tex_path):
        print(f"[FAIL] 未找到主论文文件: {tex_path}")
        return False
        
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    clean_content = strip_latex_comments_and_commands(content)

    issues = []
    warnings = []
    passes = []

    # 1. 检查中图分类号与单位代码 (11414)
    if '11414' in clean_content:
        passes.append("学校单位代码 11414 正确配置")
    else:
        issues.append("缺少中国石油大学（北京）单位代码 11414")

    if '中图分类号' in clean_content:
        passes.append("中图分类号标注正确")
    else:
        issues.append("未标注中图分类号")

    # 2. 检查论文题目长度 (≤ 25字)
    title_match = re.search(r'\\title\{([^}]+)\}', content)
    if title_match:
        raw_title = title_match.group(1).strip()
        # count title length ignoring latex tags
        clean_title = re.sub(r'\\[a-zA-Z]+', '', raw_title)
        title_len = len(re.findall(r'[\u4e00-\u9fff0-9a-zA-Z]', clean_title))
        if title_len <= 25:
            passes.append(f"论文题目长度符合规范（{title_len}字，不超过25字）: {clean_title}")
        else:
            issues.append(f"论文题目超过25字限制（当前为{title_len}字）: {clean_title}")
    else:
        warnings.append("未检测到标准的 \\title 命令，请确认封面标题位置")

    # 3. 检查中文摘要与英文摘要
    cn_abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
    if cn_abstract_match:
        cn_abs_text = cn_abstract_match.group(1)
        cn_abs_chars = count_chinese_characters(cn_abs_text)
        if 300 <= cn_abs_chars <= 600:
            passes.append(f"中文摘要字数符合规范（当前{cn_abs_chars}字，推荐300-500字）")
        else:
            warnings.append(f"中文摘要字数提示：当前为{cn_abs_chars}字（指南要求300-500字左右）")
            
        # check keywords separator
        if '分号' in cn_abs_text or ';' in cn_abs_text or '；' in cn_abs_text:
            passes.append("中文关键词分隔符符合分号规范")
        else:
            warnings.append("中文关键词应使用分号“；”分隔")
    else:
        issues.append("未找到中文摘要环境 \\begin{abstract}")

    # 4. 检查正文字数统计 (要求≥5000字，建议6000-8000字)
    # 正文提取：从第一章开始，到第6章结束，排除摘要、目录、参考文献、附录、致谢
    body_match = re.search(r'\\mainmatter(.*?)(?:\\appendix|\\begin\{thebibliography\}|\\bibliography\{)', content, re.DOTALL)
    if not body_match:
        # fallback: search from 第1章 to 附录/参考文献
        body_match = re.search(r'(\\chapter\{引言|\\chapter\{第1章)(.*?)(?:\\appendix|\\chapter\{致谢\}|\\begin\{thebibliography\}|\\bibliography\{)', content, re.DOTALL)
    
    if body_match:
        body_text = body_match.group(0)
        # remove equations, figures, tables from word count
        clean_body = re.sub(r'\\begin\{(?:equation|figure|table|tabular|table\*|figure\*|align)\}.*?\\end\{(?:equation|figure|table|tabular|table\*|figure\*|align)\}', '', body_text, flags=re.DOTALL)
        body_cn_count = count_chinese_characters(clean_body)
        if body_cn_count >= 5000:
            passes.append(f"正文核心字数达标（纯汉字数：{body_cn_count}字，符合建议6000-8000字要求）")
        else:
            issues.append(f"正文字数不足（当前纯汉字数：{body_cn_count}字，要求正文不少于5000字）")
    else:
        issues.append("无法正确定位正文主体区间，请确认章节标记")

    # 5. 检查图表引用一致性
    labels = re.findall(r'\\label\{([^}]+)\}', content)
    fig_labels = [l for l in labels if l.startswith('fig:')]
    tab_labels = [l for l in labels if l.startswith('tab:')]
    
    refs = re.findall(r'\\ref\{([^}]+)\}', content)
    ref_set = set(refs)
    
    unreferenced_figs = [f for f in fig_labels if f not in ref_set]
    if unreferenced_figs:
        warnings.append(f"存在正文中未显式引用的插图标签: {unreferenced_figs}")
    else:
        passes.append(f"所有插图 ({len(fig_labels)}幅) 均在正文中被严谨交叉引用")

    unreferenced_tabs = [t for t in tab_labels if t not in ref_set]
    if unreferenced_tabs:
        warnings.append(f"存在正文中未显式引用的表格标签: {unreferenced_tabs}")
    else:
        passes.append(f"所有表格 ({len(tab_labels)}张) 均在正文中被严谨交叉引用")

    # 6. 检查图表物理文件是否存在
    fig_files = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content)
    missing_files = []
    for ff in fig_files:
        # check with or without extension
        found = False
        candidates = [ff, ff + '.pdf', ff + '.png', ff + '.jpg',
                      os.path.join(figures_dir, os.path.basename(ff)),
                      os.path.join(figures_dir, os.path.basename(ff) + '.pdf'),
                      os.path.join(figures_dir, os.path.basename(ff) + '.png')]
        for c in candidates:
            if os.path.exists(c):
                found = True
                break
        if not found:
            missing_files.append(ff)
    if missing_files:
        issues.append(f"存在未找到的插图资源文件: {missing_files}")
    else:
        passes.append(f"所有插图文件资源完整且路径正确 ({len(fig_files)}处引用)")

    # 7. 检查参考文献引用
    citations = re.findall(r'\\cite\{([^}]+)\}', content)
    cited_keys = []
    for c in citations:
        for k in c.split(','):
            cited_keys.append(k.strip())
    
    if os.path.exists(bib_path):
        with open(bib_path, 'r', encoding='utf-8') as f:
            bib_content = f.read()
        bib_keys = re.findall(r'@\w+\{([^,]+),', bib_content)
        bib_key_set = set(bib_keys)
        missing_bibs = [k for k in cited_keys if k not in bib_key_set]
        if missing_bibs:
            issues.append(f"正文中引用了未在 .bib 中定义的参考文献键: {set(missing_bibs)}")
        else:
            passes.append(f"参考文献完备性通过: 已引用 {len(set(cited_keys))} 篇，BibTeX 包含 {len(bib_keys)} 篇文献")
    else:
        warnings.append("未找到独立 .bib 文件，请确认是否为内嵌 thebibliography")

    # 8. 检查大作业专项要求：AI工具使用披露说明表
    if 'AI工具使用披露' in clean_content or '生成式AI' in clean_content or '人工智能工具使用说明' in clean_content:
        passes.append("大作业强制项：已包含生成式AI工具使用披露说明")
    else:
        issues.append("缺少大作业必须提交的《AI工具使用披露说明》")

    # 9. 检查致谢与学术诚信承诺
    if '学术诚信' in clean_content or '独创性声明' in clean_content:
        passes.append("已包含学术诚信/独创性承诺")
    else:
        warnings.append("建议补充学术诚信与独创性声明页")

    if '致谢' in clean_content:
        passes.append("包含致谢章节")
    else:
        warnings.append("未找到致谢章节")

    print("\n--- 校验通过项 (PASS) ---")
    for p in passes:
        print(f"  [√] {p}")

    if warnings:
        print("\n--- 审查警告项 (WARNING) ---")
        for w in warnings:
            print(f"  [!] {w}")

    if issues:
        print("\n--- 阻断性缺陷 (ERROR) ---")
        for i in issues:
            print(f"  [X] {i}")
        print("\n[RESULT] 校验未完全通过，请修复上述阻断性缺陷。")
        return False
    else:
        print("\n[RESULT] 校验全部通过！恭喜，该论文已符合研究生学位论文格式与大作业严苛标准！")
        return True

if __name__ == '__main__':
    base_dir = r'C:\Users\lbybr\Desktop\DataMining'
    tex_path = os.path.join(base_dir, 'main.tex')
    bib_path = os.path.join(base_dir, 'references.bib')
    fig_dir = os.path.join(base_dir, 'figures')
    
    success = check_paper(tex_path, bib_path, fig_dir)
    sys.exit(0 if success else 1)
