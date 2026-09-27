# 数据布局

## 研究 Parquet

```text
data/parquet/<SYMBOL>/5m.parquet
data/parquet/<SYMBOL>/15m.parquet
data/parquet/<SYMBOL>/1h.parquet
```

1h 因子至少需要：`date/open/high/low/close/volume/quote_volume/trade_count/vwap`。时间必须为 UTC，`(symbol,date)` 唯一，不允许前向填充。

## Freqtrade Parquet

```text
/root/freqtrade/user_data/data/binance/futures/
  BTC_USDT_USDT-1h-futures.parquet
```

Freqtrade 文件只含 `date/open/high/low/close/volume`。使用 `tools/export_freqtrade_parquet.py` 从研究数据显式导出，避免把额外字段混入执行引擎。

## 不进入 Git 的内容

所有行情、ZIP、Parquet、模型、SQLite、日志、HTML 回测输出和缓存均被 `.gitignore` 排除。服务器数据应通过对象存储、rsync 或独立数据下载流程传输，而不是 Git。

