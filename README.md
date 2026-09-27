# Binance Quant Trial

一个把本地研究与云端 Freqtrade 执行明确分离的加密货币因子策略项目。

- 本地：保存完整 Parquet、构造 20 因子、选择 4–8 个因子、训练模型、walk-forward、生成目标仓位。
- GitHub：只保存代码、因子注册表、参数示例、测试和部署工具。
- 云服务器：从 `/root/ontime_strategy` 拉取项目；已有 `/root/freqtrade` 只负责 dry-run / live 执行。

仓库不包含行情数据、模型缓存、回测 Parquet、SQLite、日志或任何 API 密钥。

## 当前研究结论

当前透明基线使用 6 个低冗余 1h 时间序列因子。6 个月训练、每月滚动一次的 38-fold walk-forward 中，等权组合净收益约 72.5%、Sharpe 1.29、最大回撤约 -16.4%。纯 Ridge/LightGBM 的月度预测 IC 约 0.012，尚不足以替代等权；先验锚定的模型增强版本更稳健，但仍没有超过透明等权。

这些数字来自现有历史样本，不构成收益承诺。因子本身来自先前挖掘，生产前仍需新的 shadow 期。

## 目录

```text
config/                         可复制、可修改的研究参数
reference/factor_combo_strategy/
  selected_factors.csv         20 因子权威注册表
  CRYPTO_*_GUIDE.md            算子与字段精确定义
research/
  factor_combo_pipeline.py     20 因子构建、表达式求值、基础回测
  walk_forward.py              月度 walk-forward 基线
  walk_forward_v2.py           执行对齐模型、持仓审计和可视化
  settings.py                  JSON 配置加载与校验
deployment/freqtrade/user_data/
  strategies/                  可同步到服务器的策略
  config.factor-combo.example.json
tools/
  export_freqtrade_parquet.py  研究 Parquet -> Freqtrade OHLCV
  export_factor_targets.py     因子仓位 -> 历史/最新目标
  update_research_data.py      Binance 公共接口增量更新
scripts/
  cloud_bootstrap.sh           云端研究环境初始化
  sync_to_freqtrade.sh         只同步策略和示例配置
  pull_and_sync.sh             fast-forward 拉取、同步并测试
tests/                         因果算子和 walk-forward 边界测试
docs/                          架构、数据、部署和调参说明
```

## 本地研究

```powershell
git clone https://github.com/789453/binance-quant-trial.git
cd binance-quant-trial
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-research.txt
Copy-Item config\research.example.json config\research.local.json

.venv\Scripts\python -m research.walk_forward_v2 `
  --config config\research.local.json
```

行情按以下布局放置，但不会被 Git 跟踪：

```text
data/parquet/BTCUSDT/1h.parquet
data/parquet/ETHUSDT/1h.parquet
...
```

调参优先编辑 `config/research.local.json`：可更换 4–8 个因子哈希、训练窗、手续费、Ridge 正则候选、LightGBM 容量、信号平滑和调仓阈值。

## 云服务器

```bash
cd /root
git clone https://github.com/789453/binance-quant-trial.git ontime_strategy
cd /root/ontime_strategy
bash scripts/cloud_bootstrap.sh
```

脚本会建立独立的 `/root/ontime_strategy/.venv`，然后把策略文件复制到 `/root/freqtrade/user_data/strategies`。不会覆盖私密配置、数据库或已有行情。

后续更新：

```bash
cd /root/ontime_strategy
bash scripts/pull_and_sync.sh
```

## 数据接驳

将研究数据转换为 Freqtrade 可读格式：

```bash
.venv/bin/python tools/export_freqtrade_parquet.py \
  --research-data data/parquet \
  --freqtrade-user-data /root/freqtrade/user_data \
  --timeframes 1h
```

导出可回测的目标仓位历史和最新快照：

```bash
.venv/bin/python tools/export_factor_targets.py \
  --config config/research.local.json \
  --output-dir /root/freqtrade/user_data/data/factor_combo \
  --force-factors
```

`FactorComboTargetStrategy` 是 fail-closed 的接驳策略。没有目标文件、目标过期或方向不一致时拒绝新开仓。当前连续目标只能映射为入场方向和初始仓位大小，还没有实现精确的连续仓位再平衡，因此应先用于 backtest / dry-run / shadow，不应直接启用真实下单。

进一步说明：

- [系统架构](docs/ARCHITECTURE.md)
- [云端部署](docs/CLOUD_DEPLOYMENT.md)
- [数据布局](docs/DATA_LAYOUT.md)
- [因子与模型调参](docs/TUNING.md)
- [安全与发布](docs/SECURITY.md)
