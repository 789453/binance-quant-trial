# 安全与发布

- API key 只放服务器私有配置或环境变量，禁止提交。
- Binance key 禁止提现权限；研究进程不读取私钥。
- 默认 `dry_run: true`、`initial_state: stopped`。
- `FactorComboTargetStrategy` 在目标缺失、过期或方向错误时拒绝开仓。
- 推送前执行 `git status`、密钥模式扫描和 `pytest`。
- 不在 Git 历史中提交 SQLite、日志、仓位快照或包含账户余额的报告。
- 如果密钥曾进入 Git，即使随后删除，也必须撤销并重新生成；普通删除不能从历史中清除秘密。

