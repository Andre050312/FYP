# 基于 LEADv2 的多疾病 EEG 关键因素筛选

副标题：电极 × 频段 × 特征的稳定分类贡献

## 1. 研究起点

- 多篇 EEG 疾病识别论文反复使用相似的静息态时段、切窗长度和导联配置。
- 相同时段只能统一比较条件，不能说明该时段本身决定分类。
- 本项目关心哪些电极、频段和特征在跨被试测试中持续提供不可替代的信息。

## 2. 研究问题

1. 单一频段的独立判别能力有多强？
2. 从全频模型删除某个频段后，性能下降多少？
3. 哪些电极或脑区的遮挡会造成稳定性能下降？
4. LEADv2 embedding、PSD/相对功率、连接与复杂度特征中，哪一类提供额外信息？
5. 结论能否在不同折、随机种子和独立数据集上重现？

## 3. 代码选择

### LEAD-main：主模型

- 使用 `models/LEADv2.py` 和现有 P-Base 预训练权重。
- 已有时间—空间 Transformer、通道位置嵌入、采样率嵌入和受试者级损失。
- 保留 `X.dat / y.dat / meta.json` 接口和 subject-wise 划分方式。
- 需新增频段滤波、全局疾病标签映射、频段门控和系统消融。

### CenterIR：辅助方法

- CNN-BiLSTM 是抑郁严重程回归模型，不作为多疾病主干。
- 可复用 CenterIR 处理不平衡连续标签的思路，以及 SHAP 解释流程。
- 可作为 MDD 严重程回归的扩展实验。

## 4. 为什么选择 LEADv2

- LEADv2 针对头皮 EEG 预训练，与项目数据域一致。
- 可接收不同导联数、导联拓扑和采样率。
- 直接生成电极级表示，适合通道遮挡和脑区消融。
- 受试者级损失与投票符合疾病诊断的评估单位。
- 已有 AD、FTD、SZ、Depression 等数据加载器。

### LEADv2 与 LaBraM 的任务适配对比

| 维度 | LEADv2 | LaBraM |
|---|---|---|
| 原始定位 | EEG 疾病识别，原始重点为 AD | 通用 EEG/BCI 表征学习 |
| 预训练思路 | 样本级与受试者级对比学习 | tokenizer 与遮挡建模 |
| 通道建模 | 时间—空间 Transformer 与 3D 通道嵌入 | channel patch 和电极位置表示 |
| 异构适配 | 显式加入采样率嵌入，支持不同导联数和长度 | 需更严格地对齐通道、采样率和 patch |
| 受试者评估 | 原生支持受试者级损失与多数投票 | 需额外增加受试者级聚合 |
| 对本项目的工程基础 | 已有 P-Base 权重、数据加载器、微调和通道遮挡接口 | 需另外实现多疾病标签、频段消融和受试评估 |
| 本项目角色 | **主模型** | **跨模型稳定性对照** |

因此，项目使用 LEADv2 完成全部频段和电极消融。资源允许时，在 LaBraM 上重复最关键的频段与电极实验。两个模型得到一致排名时，可以降低结论依赖特定架构的风险。

## 5. 数据选择与标签

- 首轮建议：AD vs HC、MDD vs HC、SZ vs HC 三个独立任务。
- AD：ADFTD/APAVA；MDD：Mumtaz 或 Depression；SZ：ADSZ、ASEEG 或 RepOD。
- TDBRAIN 可用于后续跨诊断实验。
- ADHD/ASD 常为儿童或任务态数据，先作扩展。
- 多数据集原始数字标签必须统一映射为 `HC/AD/MDD/SZ/...`。
- 跨数据集直接多分类容易让模型学到设备和采集中心差异，首版采用共享 encoder 和疾病独立分类头。

## 6. 统一输入与预处理

- 静息态 EO 和 EC 分开报告，不与任务态直接混合。
- 去非 EEG 通道、陷波、重参考、伪影处理、通道命名对齐。
- 主窗长使用 4 s 和 8 s；delta 分析优先用 8 s。
- 在连续信号上先滤波后切窗，降低窗边界失真。
- 先分被试、再切窗。

## 7. 频段与候选因素

- broadband：0.5–45 Hz
- delta：0.5–4 Hz
- theta：4–8 Hz
- alpha：8–13 Hz
- beta：13–30 Hz
- gamma：30–45 Hz
- MDD 重点验证 Fp1/Fp2、额叶和 alpha 不对称。
- SZ 重点检查额叶—顶叶 alpha/beta 连接。
- 特征对照包含 Welch PSD、相对功率、PLV/PSI、LZC/样本熵和 LEADv2 embedding。

## 8. 第一阶段：单频段独立能力

- 在完全相同的被试划分上分别训练 broadband、delta、theta、alpha、beta 和 gamma。
- 保持窗长、通道、随机种子、优化器和 LEADv2 权重相同。
- 先冻结 encoder 做线性探测，再以小学习率微调。
- 输出每个疾病的 band-only 性能排名。

## 9. 第二阶段：不可替代性消融

- 频段：从 Full 输入分别删除 delta/theta/alpha/beta/gamma。
- 电极：单电极遮挡、对称电极对遮挡、脑区遮挡。
- 特征：比较 LEADv2 embedding、频谱、连接和复杂度特征。
- 配对贡献量为 `Full score - ablated score`。
- 证据优先级：遮挡消融 > Integrated Gradients/SHAP > attention/gate 权重。

## 10. 第三阶段：MultiBand-LEADv2

- 五个频段共享同一个 LEADv2 encoder，避免参数扩大五倍。
- 每个分支加入 band embedding，生成频段级表示。
- 频段门控融合各分支，加入稀疏约束和同一被试跨窗口一致性约束。
- 共享 encoder 后使用 AD、MDD和 SZ 独立分类头。
- 同一采集协议包含多种互斥诊断时，再增加统一多分类头。

## 11. 评估与关键因素判定

- 主指标：subject-level balanced accuracy、macro-F1 和 AUROC。
- 使用 subject-wise 5-fold 或 nested CV，并运行多个随机种子。
- 频段证据同时包含单频段性能和删除后性能下降。
- 每个结果报告 95% bootstrap 置信区间、跨折排名和入选频率。
- 至少在一个独立数据集上检查关键频段和电极排名。

## 12. 最小可行实验

1. 使用 ADSZ 或 ADFTD 打通 LEADv2 加载、冻结线性探测和受试评估。
2. 实现 broadband/theta/alpha/beta 四个输入作为烟雾测试。
3. 选定 AD、MDD、SZ 三个疾病数据集，完成全部五频段实验。
4. 完成单频段、删频段、电极遮挡和特征族对照。
5. 实现 MultiBand-LEADv2，与 PSD+SVM、EEGNet/Conformer 和原始 LEADv2 比较。

## 13. 预期产出与表述边界

- 可复现的 LEADv2 多疾病微调框架。
- 疾病 × 电极 × 频段 × 特征稳定性矩阵。
- 使用“对疾病识别的稳定判别贡献”，不宣称 EEG 因素导致疾病。
- 注意力或门控权重不等于因果重要性。
- 单数据集高准确率不等于跨中心生物标志物。

## 参考来源

- 用户提供：`EEG文章筛选.xlsx`
- LEAD 代码与论文：`LEAD-main/README.md`、`2502.01678v4.pdf`
- CenterIR 代码：`CenterIR-EEG-Depression`
