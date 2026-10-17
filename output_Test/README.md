# LEADv2 测试输出

`working/01_lead_mdd_electrode_ablation.py` 的每次运行都会在此目录生成一个新的 CSV。
文件名包含 seed 和所测电极；如同名文件已存在，自动追加 `_run02`、`_run03` 等后缀，不覆盖旧结果。
CSV 中 `none` 是未屏蔽的基线，其他行是对应电极被置零后的结果；`delta` 为基线减去屏蔽后的指标。
