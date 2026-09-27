# 系统架构

## 边界

研究进程和交易进程不共享 Python 环境，也不让研究代码直接调用私有交易 API。

```text
本地/云端研究数据
  -> causal factor library
  -> selected factor signals
  -> walk-forward / model comparison
  -> target_history.parquet + latest_targets.json
  -> Freqtrade strategy adapter
  -> dry-run / shadow / live execution
```

研究层的权威输入是 `data/parquet/<SYMBOL>/<timeframe>.parquet`。最终 20 因子只依赖原生 1h K 线中的 OHLC、quote volume、trade count 和 VWAP。Freqtrade 标准 OHLCV 只有六列，不能独立重建全部因子，因此执行层消费研究层导出的目标，而不是偷偷用 volume 替代 quote volume 或 trade count。

## 关键模块

- `factor_combo_pipeline.py`：字段、纽约时段、受限 AST 表达式、方向、信号和基础组合。
- `walk_forward_v2.py`：执行对齐标签、月度训练、模型、调仓控制、持仓审计。
- `export_factor_targets.py`：唯一的研究到执行数据契约。
- `FactorComboTargetStrategy.py`：只负责目标消费、信号过期检查和 Freqtrade 生命周期。

## 目标契约

`latest_targets.json` 包含版本、生成时间、信号 bar 和 12 个品种的 signal/target。写入采用临时文件原子替换。`target_history.parquet` 用于离线 Freqtrade 回测。

下一阶段应补充 producer/consumer 守护进程、精确连续仓位调整、交易所最小数量和名义金额量化、实际成交回写与研究目标对账。

