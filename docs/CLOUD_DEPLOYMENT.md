# 云端部署

前提：服务器已经在 `/root/freqtrade` 安装 Freqtrade，项目克隆到 `/root/ontime_strategy`。

```bash
cd /root/ontime_strategy
bash scripts/cloud_bootstrap.sh
cp config/research.example.json config/research.local.json
```

`research.local.json` 被视为机器本地参数；如果以后包含账户或容量信息，不应提交。

同步策略：

```bash
bash scripts/sync_to_freqtrade.sh
```

准备目标后先回测：

```bash
/root/freqtrade/.venv/bin/freqtrade backtesting \
  --userdir /root/freqtrade/user_data \
  -c /root/freqtrade/user_data/config.factor-combo.example.json \
  -s FactorComboTargetStrategy
```

正式 dry-run 应复制 example config 为服务器私有配置，填写 API/WebUI 信息并保持 `dry_run: true`。不要把私有配置复制回仓库。

Git 更新只允许 fast-forward：`scripts/pull_and_sync.sh` 使用 `git pull --ff-only`，避免服务器产生难以追踪的合并提交。研究修改在本地完成、测试、提交和推送；服务器原则上只拉取已验证提交。

