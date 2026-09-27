# 加密货币最终 20 因子实施说明

版本：2026-09-26  
字段公式版本：`2026-09-26-crypto-multiscale-session-v2`  
算子语义版本：`2026-09-26-v3`

## 1. 文档用途

本文档用于在另一个项目中，从原始加密货币 K 线重新构建最终 20 个因子，进一步进行因子组合和 signed long-short 策略交易。

配套文件：

- `outputs/crypto_multiscale_sessions_30000_robust20_20260926/selected_factors.csv`
- `outputs/crypto_multiscale_sessions_30000_robust20_20260926/crypto_time_series_factor_returns.html`

`selected_factors.csv` 是机器读取的权威清单。实现时应读取其中的 `expr_hash`、`expr` 和 `direction`，不要根据本文档中的展示顺序重新推断方向。

本组因子是逐币种时间序列因子，不是截面排名因子。表达式中没有 `Rank`。策略允许每个币种独立做多或做空，也允许组合在某一时刻呈现净多或净空。

## 2. 复现范围与数据依赖

### 2.1 最小数据接口

最终 20 个因子只依赖 1 小时数据。另一个项目若只复现这 20 个因子，不必加载 5 分钟和 15 分钟数据。

每行代表一个已经结束的 1h bar。最小字段如下：

| 字段 | 类型 | 单位 | 要求 |
|---|---|---|---|
| `date` | UTC timestamp | 时间 | 时区必须明确为 UTC，不能是无时区字符串 |
| `symbol` | string | 标识 | 本项目对应 `ts_code` |
| `open` | float | 报价币价格 | 有限且大于 0 |
| `high` | float | 报价币价格 | `high >= max(open, close)` |
| `low` | float | 报价币价格 | `low <= min(open, close)` |
| `close` | float | 报价币价格 | 有限且大于 0 |
| `quote_volume` | float | 报价币成交额 | 非负；原数据中的 USDT 成交额 |
| `trade_count` | float/int | 笔数 | 非负 |
| `vwap` | float | 报价币价格 | 有限且大于 0 |

原项目加载器还读取 `volume`、`taker_buy_volume`、`taker_sell_volume` 和 `taker_buy_ratio`，用于更大的候选字段池；它们不是最终 20 因子的必需依赖。

### 2.2 数据排序和完整性

处理前必须：

1. 按 `symbol, date` 升序排序。
2. 确保 `(symbol, date)` 唯一。
3. 使用统一的整点 UTC 时间栅格。
4. 只在 bar 结束后计算该 bar 的字段。
5. 不前向填充价格、成交额或因子值。
6. 将正负无穷替换为 `NaN`。

最终复杂因子的最长链条包含 720h 滚动相关、滞后差分、第二层 720h 标准化，再加策略层 168h 标准化。建议实盘或回测开始时间之前至少预留 90 天 1h 数据作为 warm-up。

### 2.3 多周期扩展接口

本轮研究候选池同时使用过 15m 和 5m 数据，包括日内实现波动、跳跃占比、路径效率、流量趋势、成交集中度和 5m–15m 差值。它们参与了 30,000 个表达式的竞争，但未进入最终 20。因此：

- 复现最终 20：只需要 1h。
- 继续挖掘或替换因子：再接入 15m 和 5m。
- 不要用 5m/15m 数据改写本文档定义的 1h 基础字段，否则无法与原结果对账。

## 3. 通用数值约定

### 3.1 安全除法

所有比例采用：

```text
safe_div(a, b) = a / b,  当 |b| >= 1e-9
                 NaN,    当 |b| < 1e-9
```

不允许把零分母结果替换为 0。

### 3.2 滚动窗口

所有窗口单位均为 1h bar：

- `3` = 3 小时
- `6` = 6 小时
- `12` = 12 小时
- `24` = 1 天
- `72` = 3 天
- `168` = 7 天
- `336` = 14 天
- `720` = 30 天

滚动运算只能使用当前时点及历史数据。本文档明确要求 `shift(1)` 的位置必须先滞后再滚动。

### 3.3 缺失值

- 原始依赖缺失时，不生成替代值。
- `TsDelta` 任一端缺失则结果缺失。
- 滚动相关仅使用左右两列同时有限的成对观测。
- 滚动标准差使用总体标准差，即 `ddof=0`。
- 任何最终无穷值都替换为 `NaN`。

## 4. 最终因子所需基础字段

以下公式均按单个 `symbol` 独立计算，除纽约时段同小时基线外，不跨币种混合数据。

### 4.1 价格和收益

```text
ret_1h[t]  = close[t] / close[t-1]  - 1
ret_4h[t]  = close[t] / close[t-4]  - 1
ret_24h[t] = close[t] / close[t-24] - 1

oc_ret[t]  = (close[t] - open[t]) / open[t]
vwap_bias[t] = (close[t] - vwap[t]) / vwap[t]
hl_range[t]  = (high[t] - low[t]) / open[t]
```

### 4.2 波动水平和期限结构

```text
realized_vol_4h[t]  = std(ret_1h[t-3:t], ddof=0),  min_periods=3
realized_vol_24h[t] = std(ret_1h[t-23:t], ddof=0), min_periods=12

volatility_term_slope[t]
    = safe_div(realized_vol_4h[t], realized_vol_24h[t]) - 1
```

### 4.3 流动性和交易效率

```text
quote_volume_log[t] = log(1 + max(quote_volume[t], 0))
trade_count_log[t]  = log(1 + max(trade_count[t], 0))

liquidity_efficiency[t]
    = safe_div(abs(ret_1h[t]), trade_count_log[t])

amihud_raw[t]
    = safe_div(abs(ret_1h[t]), quote_volume[t])

amihud_log[t]
    = log(1 + amihud_raw[t] * 1e9)
```

`trade_count_shock_24h` 使用包含当前值的普通滚动基线：

```text
mu[t]    = mean(trade_count_log[t-23:t]), min_periods=12
sigma[t] = std(trade_count_log[t-23:t], ddof=0), min_periods=12

trade_count_shock_24h[t]
    = safe_div(trade_count_log[t] - mu[t], sigma[t])
```

### 4.4 纽约本地交易时段

先将 UTC 时间转换到 IANA 时区 `America/New_York`。必须使用支持夏令时的时区库，不能使用固定的 UTC-5 或 UTC-4。

按纽约本地时间划分三个互斥的 8 小时时段：

```text
overnight: 00:00 <= hour < 08:00
day:       08:00 <= hour < 16:00
evening:   16:00 <= hour < 24:00
```

```text
us_overnight_flag[t] = 1 if NY hour < 8 else 0
```

时段键定义为：

```text
session_key = (symbol, NY local calendar date, session_code)
```

注意，凌晨、白天和晚间是同一个本地日期下的三个独立区段；这里没有把凌晨归入前一交易日。

```text
session_cumulative_return[t]
    = close[t] / first_open_of_current_session - 1
```

### 4.5 同纽约小时季节性校正

`session_volume_surprise`、`session_volatility_surprise` 和 `session_illiquidity_surprise` 均按 `(symbol, NY local hour)` 分组。

对源序列 `x`，严格使用当前时点以前的同小时观测：

```text
history = x.shift(1)
mu      = rolling_mean(history, 60, min_periods=10)
sigma   = rolling_std(history, 60, min_periods=10, ddof=0)
surprise = safe_div(x - mu, sigma)
```

三个字段的源序列分别是：

| 字段 | 源序列 |
|---|---|
| `session_volume_surprise` | `quote_volume_log` |
| `session_volatility_surprise` | `hl_range` |
| `session_illiquidity_surprise` | `amihud_log` |

这里的 `60` 表示过去 60 个同纽约小时观测，约等于过去 60 天，而不是过去 60 个连续小时。`shift(1)` 是防止未来信息和当前值污染基线的关键步骤。

## 5. 因子表达式算子

最终 20 只用到以下六个算子。

### 5.1 `Sub(x, y)`

```text
Sub(x, y)[t] = x[t] - y[t]
```

### 5.2 `Mul(x, y)`

```text
Mul(x, y)[t] = x[t] * y[t]
```

### 5.3 `TsDelta(x, w)`

```text
TsDelta(x, w)[t] = x[t] - x[t-w]
```

这是绝对差，不是百分比变化。

### 5.4 `TsEMA(x, w)`

与以下 Pandas 语义一致：

```python
series.ewm(span=w, min_periods=w, adjust=False).mean()
```

平滑系数为 `2 / (w + 1)`。前 `w-1` 个有效期不输出。

### 5.5 `TsCorr(x, y, w)`

对每个币种独立计算长度为 `w` 的 Pearson 时间序列相关：

```text
TsCorr(x, y, w)[t]
    = corr(x[t-w+1:t], y[t-w+1:t])
```

精确语义：

- 必须先经过 `w` 个时间位置，即 `t >= w-1`。
- 只使用 `x`、`y` 同时有限的成对观测。
- 有效成对观测数至少为 `max(2, floor(w/2))`。
- 任一侧窗口内方差接近 0 时输出 `NaN`。
- 不做截面相关，不混合不同币种。

### 5.6 `TsZScore(x, w)`

```text
TsZScore(x, w)[t]
    = (x[t] - rolling_mean(x, w)[t])
      / rolling_std(x, w, ddof=0)[t]
```

精确语义：

- 必须先经过 `w` 个时间位置。
- 至少需要 `floor(w/2)+1` 个有限观测。
- 当前 `x[t]` 必须有限。
- 标准差小于等于 `1e-12` 时输出 `NaN`。
- 均值和标准差窗口包含当前值。

## 6. 最终 20 个因子

表中顺序按验证期净累计收益排列，仅用于阅读。机器实现必须以 CSV 中的 `expr_hash`、`expr`、`direction` 为准。

|序号|方向|表达式|发现收益|发现 Sharpe|验证收益|验证 Sharpe|
|---:|---:|---|---:|---:|---:|---:|
|1|-1|`TsZScore(TsDelta(TsCorr($ret_24h,$realized_vol_24h,720),12),720)`|25.44%|0.632|94.97%|2.055|
|2|-1|`TsZScore(TsDelta(TsCorr($ret_24h,$realized_vol_24h,720),6),720)`|18.30%|0.512|77.89%|1.836|
|3|-1|`TsZScore(TsDelta(TsCorr($ret_24h,$session_volatility_surprise,720),12),720)`|4.64%|0.220|64.06%|1.597|
|4|-1|`TsZScore(TsDelta(TsCorr($ret_24h,$realized_vol_24h,720),3),720)`|11.14%|0.366|56.19%|1.480|
|5|-1|`TsZScore(TsDelta(TsCorr($ret_4h,$session_volatility_surprise,720),24),720)`|29.53%|0.675|54.69%|1.354|
|6|+1|`TsZScore(TsDelta(TsCorr($volatility_term_slope,$session_volume_surprise,720),24),720)`|32.74%|0.830|42.33%|1.285|
|7|-1|`TsZScore(TsCorr($session_volatility_surprise,$us_overnight_flag,72),72)`|0.96%|0.144|35.93%|1.042|
|8|-1|`TsZScore(TsDelta(TsCorr($session_cumulative_return,$session_volatility_surprise,720),24),720)`|27.61%|0.654|29.78%|0.875|
|9|-1|`TsZScore(TsDelta(TsCorr($ret_4h,$session_volume_surprise,720),24),720)`|30.36%|0.689|25.12%|0.752|
|10|-1|`TsZScore(TsDelta(TsCorr($ret_24h,$session_volume_surprise,720),12),720)`|5.62%|0.243|24.53%|0.775|
|11|+1|`TsZScore(Sub(TsEMA($session_volatility_surprise,24),TsEMA($session_volatility_surprise,336)),336)`|2.53%|0.192|21.04%|0.622|
|12|-1|`TsZScore(TsDelta(TsCorr($ret_4h,$liquidity_efficiency,720),24),720)`|48.80%|0.973|20.13%|0.648|
|13|-1|`TsZScore(Sub(TsCorr($ret_4h,$session_illiquidity_surprise,24),TsCorr($ret_4h,$session_illiquidity_surprise,72)),72)`|5.32%|0.242|14.37%|0.646|
|14|-1|`TsZScore(TsDelta(TsCorr($session_cumulative_return,$session_volume_surprise,720),24),720)`|16.64%|0.459|14.04%|0.506|
|15|-1|`TsZScore(TsDelta(TsCorr($ret_1h,$session_volatility_surprise,720),24),720)`|21.41%|0.551|13.69%|0.512|
|16|-1|`TsZScore(TsDelta(TsCorr($oc_ret,$session_volatility_surprise,720),24),720)`|21.33%|0.549|13.66%|0.511|
|17|+1|`Mul(TsZScore(TsEMA($volatility_term_slope,24),720),TsZScore(TsEMA($session_volatility_surprise,24),720))`|4.42%|0.213|9.97%|0.421|
|18|+1|`TsZScore(Sub(TsEMA($session_volatility_surprise,12),TsEMA($session_volatility_surprise,336)),336)`|2.08%|0.175|7.15%|0.334|
|19|+1|`TsZScore(TsDelta(TsCorr($vwap_bias,$trade_count_shock_24h,72),12),72)`|18.18%|0.521|4.01%|0.251|
|20|-1|`TsZScore(TsDelta(TsCorr($ret_1h,$liquidity_efficiency,720),24),720)`|17.33%|0.467|0.18%|0.152|

`direction=-1` 表示将表达式输出乘以 -1 后使用。方向已经在发现期冻结。另一个项目不得根据验证期、holdout 或实盘近期收益重新翻转方向。

## 7. 从表达式到可交易信号

### 7.1 单因子标准策略

设表达式输出为 `f[k,t,s]`，其中 `k` 为因子，`t` 为小时，`s` 为币种。设 CSV 中方向为 `d[k]`。

原回测不是直接拿表达式的外层 Z-score 当仓位，而是在策略层再次按每个币种做 168h 标准化：

```text
z[k,t,s] = TsZScore(f[k,:,s], 168)[t]

target[k,t,s]
    = clip(z[k,t,s] * d[k], -2, 2) / 2 / N
```

`N` 为同时交易的币种数量，本轮是 12。随后构造 4 个重叠持仓批次，等价于：

```text
smoothed_target[k,t,s] = rolling_mean(target[k,:,s], 4)[t]
```

原实现的 `rolling_mean(..., 4)` 在至少 3 个有限值时输出。

执行仓位采用：

```text
position[k,t,s] = smoothed_target[k,t-2,s]
```

这里必须保留两根 1h bar 的位移。它来自原框架 `entry_lag=1` 与收益索引约定的组合。若另一个回测引擎采用不同的 bar-open/bar-close 对齐方式，应以“不使用尚未结束的 bar，且与原项目 `t-2` 仓位一致”为验收标准。

单小时组合收益：

```text
asset_return[t,s] = close[t,s] / close[t-1,s] - 1

gross_return[t] = sum_s(position[t,s] * asset_return[t,s])

turnover[t] = sum_s(abs(position[t,s] - position[t-1,s]))

net_return[t] = gross_return[t] - turnover[t] * 4 / 10000
```

该仓位是 signed long-short。不要改成“选正信号做多，其余空仓”。

### 7.2 多因子组合建议

最终 20 中存在多个相近的 720h 相关变化因子。直接等权会让同一类结构重复获得权重。建议先分组，再组合。

建议分为以下六组：

1. 收益–实现波动相关变化：1、2、4。
2. 收益–时段波动惊喜：3、5、15、16。
3. 收益–时段成交量惊喜：9、10。
4. 时段累计状态：7、8、14。
5. 流动性和成交效率：12、13、19、20。
6. 平滑状态和跨尺度状态：6、11、17、18。

推荐的透明基线：

```text
oriented_signal[k] = clip(direction[k] * TsZScore(factor[k], 168), -2, 2) / 2

group_signal[g] = mean(oriented_signal[k] for k in group g)

combo_signal = mean(group_signal[g] for all active groups)

position[t,s] = rolling_mean(combo_signal[:,s], 4)[t-2] / N
```

这样先在相关模板内部平均，再在经济维度之间平均，避免三个近似的 `ret_24h`–`realized_vol_24h` 变体获得三倍权重。

进阶组合可在发现期数据上计算因子策略收益协方差，使用带收缩的逆波动或风险平价权重。约束建议：

- 权重非负，因为方向已经由 `direction` 固定。
- 单因子权重不超过 15%。
- 单经济分组权重不超过 30%。
- 权重只在预设周期更新，例如每 30 天一次。
- 权重估计不得读取验证期之后的数据。
- 若样本不足，回退到分组等权，不要静默回退到纯多头。

### 7.3 两因子配对复现

配套 HTML 中的两因子组合采用另一种明确规则。对一对因子 `a,b`：

```text
pair_raw = 0.5 * (direction[a] * factor[a] + direction[b] * factor[b])
pair_z   = TsZScore(pair_raw, 168)
target   = clip(pair_z, -2, 2) / 2 / N
```

随后同样做 4h 滚动持仓、`t-2` 执行和 4 bps 换手成本。若目标是复现配对 HTML，应使用该规则，而不是先分别 Z-score 后再相加。

## 8. 推荐的软件模块边界

建议另一个项目按以下接口拆分：

```text
raw_loader
  -> validate_hourly_bars
  -> build_base_features
  -> build_new_york_session_features
  -> evaluate_expression_tree
  -> orient_and_normalize_signals
  -> combine_signals
  -> construct_positions
  -> apply_execution_and_costs
  -> report_and_audit
```

不要把字段构造、表达式计算和回测仓位写在一个函数中。至少保留：

- `features`: 只负责无未来信息的字段矩阵。
- `operators`: 只负责明确的数组运算和 NaN 语义。
- `expressions`: 从 CSV 解析表达式并调用算子。
- `portfolio`: 方向、标准化、持仓平滑、延迟和成本。
- `audit`: 时间戳、覆盖率、前缀不变性和黄金样本对账。

## 9. 一致性验收

### 9.1 静态检查

- CSV 必须恰好包含 20 条已选因子。
- `expr_hash` 必须唯一。
- 所有依赖字段必须属于本文档第 4 节的字段集合。
- 所有表达式只使用第 5 节定义的算子。
- `direction` 只能为 `+1` 或 `-1`。

### 9.2 前缀不变性

对任意截止时间 `T`：

1. 用截止到 `T` 的数据计算全部字段和因子。
2. 用包含 `T` 之后数据的完整样本再次计算。
3. 两次结果在 `<=T` 的区域必须一致，允许相同位置同时为 `NaN`。

若不一致，通常表示同小时季节性基线遗漏了 `shift(1)`、时间排序错误，或引入了双向填充。

### 9.3 黄金样本对账

从原项目导出一段至少 120 天、覆盖夏令时切换的字段和因子矩阵。另一个项目应逐层对账：

1. `ret_1h`、`ret_4h`、`ret_24h`。
2. `realized_vol_24h` 和 `volatility_term_slope`。
3. 三个 `session_*_surprise`。
4. 每个表达式的有限值掩码。
5. 表达式数值和方向调整后数值。
6. 168h 策略 Z-score、4h 平滑仓位和 `t-2` 执行仓位。
7. 无成本收益、换手和含成本收益。

建议容差：

```text
绝对误差 <= 1e-10，或相对误差 <= 1e-8
```

对滚动相关和标准差，应优先比较有限值掩码，再比较数值。不同库在接近零方差时可能产生极小差异，但不得改变有效/无效窗口的判断。

### 9.4 时区测试

至少测试美国春季和秋季夏令时切换周：

- 每个 UTC 小时映射到正确的纽约本地小时。
- 三个时段 flag 互斥且和为 1。
- 同纽约小时基线在 DST 切换后仍按本地小时分组。
- 不产生重复 `(symbol, UTC timestamp)`。

## 10. 常见错误

- 把所有 720h 因子当作截面因子进行每小时横截面排名。
- 根据验证期收益重新翻转 `direction`。
- 在 `session_*_surprise` 中遗漏 `shift(1)`。
- 用固定 UTC 偏移替代 `America/New_York`。
- 把 `720h` 误写成 720 天。
- `TsDelta` 使用百分比变化而不是绝对差。
- 滚动标准差使用 `ddof=1`。
- 把缺失值、零分母或零方差窗口强制填成 0。
- 直接在表达式输出上建仓，遗漏策略层第二次 168h 标准化。
- 使用 `t-1` 仓位而不是原框架的 `t-2` 对齐。
- 把负信号设为空仓，导致策略退化为纯多头。
- 对 20 个高度相关因子简单等权，不先做经济分组或相关性约束。

## 11. 版本与证据

- 最终 20 CSV SHA-256：`D04EAE5C2A5118062DB4069AC3EBE0E6010BA572C152DF686F6B140BC8F64C70`
- 累计收益 HTML SHA-256：`9577AD770595F6D0D4F008328447A584A30B80766C6137919DD091136B093D3C`
- 实验 manifest SHA-256：`2987CE6BE65420D6DBC0C9CF3ACBB7F2680E0FF5A8F4F1F2EC3DCDDD4E8F05C3`

若 CSV、HTML 或 manifest 的哈希变化，应视为新的研究版本，重新进行黄金样本和前缀不变性验证。
