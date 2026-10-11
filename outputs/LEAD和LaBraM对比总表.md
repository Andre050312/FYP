# LEADv2 与 LaBraM 公开数据集对比

更新日期：2026-10-10

## 四任务结果总表

全部数值为测试结果，单位为 %。患者级准确率统一采用片段多数投票。

| 任务 / 数据集 | 模型 | 片段准确率 | 片段 AUROC | 患者级准确率 | 评估方案 |
|---|---|---:|---:|---:|---|
| AD / ADFTD-RS | LEADv2 | 79.34 | 89.59 | 85.71（6/7） | 200 Hz，固定划分 |
| AD / ADFTD-RS | LaBraM | 73.25 | 81.89 | 85.71（6/7） | 200 Hz，固定划分 |
| FTD / ADFTD-RS | LEADv2 | 92.74 | 98.01 | 100.00 | 固定划分，实际采样率待补证 |
| FTD / ADFTD-RS | LaBraM | 83.80 | 94.98 | 100.00（6/6） | 200 Hz，seed 41 |
| MDD / Mumtaz | LEADv2 | 90.15 ± 5.55 | 98.36 ± 1.42 | 93.33 ± 8.16 | 200 Hz，五次 MCCV，seed 41–45 |
| MDD / Mumtaz | LaBraM | 91.79 ± 5.27 | 98.69 ± 1.45 | 96.67 ± 6.67 | 200 Hz，五次 MCCV，seed 41–45 |
| SZ / RepOD | LEADv2 | 61.79 ± 6.73 | 62.64 ± 10.37 | 56.67 ± 8.16 | 200 Hz，五次 MCCV |
| SZ / RepOD | LaBraM | 70.93 ± 9.33 | 79.45 ± 3.86 | 76.67 ± 17.00 | 200 Hz，五次 MCCV |

MDD 与 SZ 的“±”为五次结果的总体标准差，不是置信区间；MCCV 是五次重复随机被试划分，不是互不重叠的五折交叉验证。AD 和 FTD 仍为单次固定划分。FTD 片段结果尚待核实 LEAD 采样率。MDD 每次仅测试 6 名被试，不同次测试的被试可能重复，因此五次结果不能视为 30 名独立患者。LEAD 结果文件未保存逐人预测和实际测试 ID；虽然两边加载器按相同 seed 设计划分，仍需补齐运行记录以严格确认逐人配对。LaBraM 训练日志中的梯度范数曾出现 Infinity/NaN，数值稳定性需另行排查。

## MDD 五次重复划分与早期单次结果

| seed | LEADv2 患者级多数投票 | LaBraM 患者级多数投票 |
|---:|---:|---:|
| 41 | 6/6 | 6/6 |
| 42 | 6/6 | 6/6 |
| 43 | 5/6 | 5/6 |
| 44 | 6/6 | 6/6 |
| 45 | 5/6 | 6/6 |

早期固定划分结果另行保留：LEADv2 片段准确率 89.91%、片段 AUROC 98.78%、患者级多数投票 6/6；LaBraM 分别为 83.14%、93.57%、5/6。固定划分的随机划分种子为 42，而 MCCV seed 41 使用 41；即使训练 seed 同为 41，两者测试被试也不同，不能按同一次划分直接比较。早期固定划分中 LaBraM 的平均概率投票为 6/6，但本表统一采用多数投票。

数据来源与逐任务说明：[LEADv2与LaBraM公开数据集对比报告.md](../LEADv2与LaBraM公开数据集对比报告.md)

MDD 原始来源：

- [LEAD 五次 MCCV 结果](C:/Users/ROG/Desktop/LEAD/results/LEADv2/finetune/LEADv2/P-Base-F-MDD-Mumtaz-200Hz-MCCV5/results.txt)
- LaBraM 五次独立评估：[seed 41](C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_MCCV_seed41_eval/subject_metrics.json)、[42](C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_MCCV_seed42_eval/subject_metrics.json)、[43](C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_MCCV_seed43_eval/subject_metrics.json)、[44](C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_MCCV_seed44_eval/subject_metrics.json)、[45](C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_MCCV_seed45_eval/subject_metrics.json)
- 早期固定划分：[LEAD 结果](C:/Users/ROG/Desktop/LEAD/results/LEADv2/finetune/LEADv2/P-Base-F-MDD-Mumtaz-200Hz-Fixed/results.txt)、[LaBraM 独立评估](C:/Users/ROG/Desktop/LaBraM-main/outputs/LaBraM_MDD-Mumtaz_200Hz_seed41/subject_metrics.json)
