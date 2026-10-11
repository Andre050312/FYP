from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
report=root/'LEADv2与LaBraM公开数据集对比报告.md'
lead='C:/Users/ROG/Desktop/LEAD/results/LEADv2/finetune/LEADv2/P-Base-F-MDD-Mumtaz-200Hz-Fixed/results.txt'
lab='C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_seed41/subject_metrics.json'
pred='C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_seed41/subject_predictions.csv'
m=json.loads(Path(lab).read_text())
assert m['test_subject_count']==6 and m['test_window_count']==1714
assert round(m['window_accuracy']*100,2)==83.14
assert round(m['window_auroc']*100,2)==93.57
table='''## 四任务结果总表

全部数值为测试结果，单位为 %。患者级准确率统一采用片段多数投票。

| 任务 / 数据集 | 模型 | 片段准确率 | 片段 AUROC | 患者级准确率 | 评估方案 |
|---|---|---:|---:|---:|---|
| AD / ADFTD-RS | LEADv2 | 79.34 | 89.59 | 85.71（6/7） | 200 Hz，固定划分 |
| AD / ADFTD-RS | LaBraM | 73.25 | 81.89 | 85.71（6/7） | 200 Hz，固定划分 |
| FTD / ADFTD-RS | LEADv2 | 92.74 | 98.01 | 100.00 | 固定划分，实际采样率待补证 |
| FTD / ADFTD-RS | LaBraM | 83.80 | 94.98 | 100.00（6/6） | 200 Hz，seed 41 |
| MDD / Mumtaz | LEADv2 | 89.91 | 98.78 | 100.00 | 200 Hz 标记运行，固定划分，seed 41 |
| MDD / Mumtaz | LaBraM | 83.14 | 93.57 | 83.33（5/6） | 200 Hz，固定划分，seed 41 |
| SZ / RepOD | LEADv2 | 61.79 ± 6.73 | 62.64 ± 10.37 | 56.67 ± 8.16 | 200 Hz，五次 MCCV |
| SZ / RepOD | LaBraM | 70.93 ± 9.33 | 79.45 ± 3.86 | 76.67 ± 17.00 | 200 Hz，五次 MCCV |

SZ 的“±”为五次结果的总体标准差，不是置信区间。其他任务是单次固定划分，不具有相同的重复实验支持。FTD 片段结果尚待核实 LEAD 采样率。MDD 的 LEAD 结果未保存逐人预测及测试片段数，不能仅凭同一加载器的划分逻辑断定旧运行的实际测试 ID 完全一致。LaBraM MDD 平均概率投票为 100%，此处统一采用多数投票的 83.33%。
'''
section=f'''### 3.4 Mumtaz：MDD vs HC，固定被试划分

LEADv2 使用 `P-Base-F-MDD-Mumtaz-200Hz-Fixed` 完整运行，模型设置记录 seed 41，单次结果。LaBraM 独立评估记录采样率 200 Hz、seed 41，测试集为 6 名被试、1,714 个片段。加载器针对闭眼 EEG 的 MDD 与健康对照分类，固定划分逻辑使用 random.Random(42)，训练 seed 41 不等于划分随机种子。

| 测试指标 | LEADv2 | LaBraM | 当前比较 |
|---|---:|---:|---|
| 片段准确率 | 89.91% | 83.14% | LEADv2 高 6.77 个百分点 |
| 片段 AUROC | 98.78% | 93.57% | LEADv2 高 5.21 个百分点 |
| 患者级准确率，多数投票 | 100.00% | 83.33%（5/6） | LEADv2 高 16.67 个百分点 |

LaBraM 的平均概率投票准确率为 100%（6/6），但多数投票只判对 5/6，不能将这两个指标混用。被试 123 的平均预测概率为 0.50763，平均概率投票判为 MDD，但片段多数投票判为 HC。表内与其他任务统一采用多数投票口径。测试被试 ID 为 1、4、23、111、123、130。LEAD 结果文件没有保存实际测试 ID、测试片段数及逐人预测，100% 暂保留原始百分比，不额外声称已验证为同一组 6/6。需补齐记录后再作严格配对比较，并进行多次被试划分复核。

来源：[LEAD MDD 结果]({lead})、[LaBraM MDD 独立评估]({lab})、[LaBraM MDD 逐人预测]({pred})。

'''
s=report.read_text(encoding='utf-8').replace('2026-10-07','2026-10-10',1)
s=s.replace('## 二、模型与本次比较的定位',table+'\n## 二、模型与本次比较的定位')
s=s.replace('## 四、可比性、局限与不能下的结论',section+'## 四、可比性、局限与不能下的结论')
s=s.replace('这些结果支持继续研究', 'MDD-Mumtaz 的单次固定划分中，LEADv2 片段准确率、片段 AUROC 和多数投票患者级准确率较高。LaBraM 的患者级多数投票为 5/6，平均概率投票为 6/6，须严格区分口径。LEAD MDD 的实际测试 ID 仍需补齐。\n\n这些结果支持继续研究',1)
s=s.replace('AD 测试 7 人，FTD 测试 6 人，RepOD 每次测试 6 人','AD 测试 7 人，FTD 测试 6 人，LaBraM MDD 测试 6 人，RepOD 每次测试 6 人')
s=s.replace('SZ 来自 RepOD。','SZ 来自 RepOD，MDD 来自 Mumtaz。')
s=s.replace('在 AD 与 FTD 任务上使用相同','在 AD、FTD 与 MDD 任务上使用相同')
report.write_text(s,encoding='utf-8')
(root/'outputs'/'LEAD和LaBraM对比总表.md').write_text('# LEADv2 与 LaBraM 公开数据集对比\n\n更新日期：2026-10-10\n\n'+table+'\n数据来源与逐任务说明：['+report.name+'](../'+report.name+')\n\nMDD 原始来源：\n\n- [LEAD 结果]('+lead+')\n- [LaBraM 独立评估]('+lab+')\n- [LaBraM 逐人预测]('+pred+')\n',encoding='utf-8')

b=root/'.build_lead_compare'/'build.py'
t=b.read_text(encoding='utf-8-sig')
t=t.replace("'研究重点：电极与频段的贡献能否跨划分保持稳定。'", "'MDD：LEADv2 指标较高，多数投票 100% 对 83.33%。'")
t=t.replace("['RepOD','SZ vs HC", "['Mumtaz','MDD vs HC\\n抑郁症 / 健康对照','固定划分，LaBraM 测试 6 人'],['RepOD','SZ vs HC")
needle="add('SZ：LaBraM 的五次平均表现更高'"
new="""add('MDD：LEADv2 的当前指标较高',rows=[['指标','LEADv2','LaBraM'],['片段准确率','89.91%','83.14%'],['片段 AUROC','98.78%','93.57%'],['患者级多数投票','100.00%','83.33%（5/6）']],widths=[250,191,191],foot='Mumtaz 固定划分，200 Hz 标记运行，训练 seed 41。LEAD 测试 ID 尚待补齐。',note='LaBraM 测试 6 人、1,714 片段。平均概率投票为 100%（6/6），本表统一采用多数投票。被试 123 平均概率 0.50763，但多数投票判错。LEAD 结果未保存测试 ID、片段数及逐人预测，不能直接声称已验证同一组 6/6。固定划分加载器使用 random.Random(42)。来源："""+lead+'；'+lab+'；'+pred+"""')
"""
t=t.replace(needle,new+needle)
summary="""add('四任务结果总表',rows=[['任务','片段准确率','片段 AUROC','患者级准确率'],['AD','79.34 / 73.25','89.59 / 81.89','85.71 / 85.71'],['FTD*','92.74 / 83.80','98.01 / 94.98','100 / 100'],['MDD','89.91 / 83.14','98.78 / 93.57','100 / 83.33'],['SZ 均值','61.79 / 70.93','62.64 / 79.45','56.67 / 76.67']],widths=[116,172,172,172],foot='每格依次为 LEADv2 / LaBraM，单位 %。患者级采用多数投票。FTD* 配置待补证。',note='SZ 为五次 MCCV 均值，标准差见 SZ 结果页。其他任务为单次固定划分。MDD 的 LEAD 实际测试 ID 尚待补齐。')
"""
t=t.replace("add('比较结果的适用范围'",summary+"add('比较结果的适用范围'")
t=t.replace('为 AD 与 FTD 固定多次被试划分','为 AD、FTD 与 MDD 固定多次被试划分').replace('报告更新：2026-10-07','报告更新：2026-10-10')
old="if d['rows']: table(tree,3,d['rows'],d['widths'],102 if i==3 else 110,48 if i==3 else 58,17 if i in (3,4) else 18)"
new="if d['rows']: table(tree,3,d['rows'],d['widths'],102 if len(d['rows'])==5 else 110,46 if len(d['rows'])==5 else 58,17 if len(d['rows'])==5 else 18)"
assert old in t
t=t.replace(old,new)
b.write_text(t,encoding='utf-8')
print('Updated report, summary table, and PPT authoring source.')
