"""Attach independently audited baseline scores to the current report."""
from pathlib import Path
import csv
import re


def attach(root):
    root=Path(root);source=root/'brier_audit/comparison_1s.csv'
    if not source.exists():return
    with source.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    names={'learned_seed_2026':'DeepONet–Hopf，seed2026','zero_noise':'DeepONet–Hopf，零噪声'}
    cells=[];markdown=[]
    for r in rows:
        name=r.get('display_name') or names.get(r['method'],r['method'])
        cells.append(f"<tr><td>{name}</td><td>{float(r['brier']):.9f}</td></tr>")
        markdown.append(f"| {name} | {float(r['brier']):.9f} |")
    begin='<!-- BEGIN BRIER AUDIT -->';end='<!-- END BRIER AUDIT -->'
    section=begin+'''<section id="brier-audit-section"><h2>Brier 独立核查与基线</h2><p>全部使用相同的 8,234 个评分起点。原始参考标签、完整随访掩码、逐路径概率与 pooled 汇总已独立核对一致。阈值与两次确认不进入 Brier。</p><table><thead><tr><th>方法</th><th>pooled Brier</th></tr></thead><tbody>'''+''.join(cells)+'''</tbody></table><p>生存模型总 Brier 略低于 DeepONet。纯峰时钟是将硬规则 0/1 代入该公式，并非校准概率模型。上述机制基线对应 1 秒目标，不能作为 2/3/5 秒基线分数。<a href="brier_audit/index.html">详细审计、正负例分解、汇总口径与 CSV</a></p></section>'''+end
    page=root/'index.html';text=page.read_text(encoding='utf-8')
    text=re.sub(re.escape(begin)+'.*?'+re.escape(end),'',text,flags=re.S)
    anchor='<h2>21 条记录</h2>'
    text=text.replace(anchor,section+anchor,1) if anchor in text else text.replace('</body>',section+'</body>',1)
    page.write_text(text,encoding='utf-8')
    report=root/'REPORT_中文.md';text=report.read_text(encoding='utf-8').split(begin)[0].rstrip()
    text+='\n\n'+begin+'\n## Brier 独立核查与机制基线\n\n所有方法使用相同的 8,234 个评分起点。独立重算与主表完全一致。\n\n| 方法 | pooled Brier |\n|---|---:|\n'+'\n'.join(markdown)+'\n\n纯峰时钟为 0/1 硬规则评分；其余为概率分数。当前生存模型的 pooled Brier 略低；DeepONet 的正例误差较低、负例误差较高，不能只按有利项宣称全面优越。基线均为 1 秒目标。细节见 brier_audit/REPORT_中文.md。\n'+end+'\n'
    report.write_text(text,encoding='utf-8')


if __name__=='__main__':attach(Path(__file__).resolve().parent)
