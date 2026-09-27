# 因子与模型调参

复制 `config/research.example.json` 为 `config/research.local.json`。

## 因子

`selected_hashes` 接受 4–8 个唯一哈希。候选必须来自 `reference/factor_combo_strategy/selected_factors.csv`。不要修改表达式参数后继续沿用原哈希，也不要根据最终留出期重新翻转 `direction`。

建议流程：训练期内计算相关矩阵；设置相关性上限；按发现期或内部训练期指标选择；冻结清单；在后续月份 walk-forward。每次实验保留 config 和 manifest。

## Walk-forward

- `train_months`：默认 6，可比较 6/12/18，但比较本身会消耗验证集。
- `step_months`：生产建议 1。
- `fee_bps`：至少覆盖 taker 费率；另做 2/4/8/12 bps 压力测试。

## 模型

Ridge 的 `alphas` 在每个训练窗最后一个月内部验证。LightGBM 应保持浅树、较大叶节点样本数和 early stopping。不要只按 RMSE 选模型；必须同时检查 IC、换手、成本后收益、月度稳定性、多空方向和币种集中度。

## 执行

`z_window`、`smooth_span`、`neutral_zone` 和 `position_deadband` 直接决定换手。调参时同时报告 gross return 和 cost drag。自适应模型是卫星仓位，`adaptive_max_model_weight` 默认 25%，不应在缺少独立 shadow 结果时提高。

