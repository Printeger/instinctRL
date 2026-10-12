# Coordination 工作区指令

现行入口是 [ACTIVE_WORKFLOW.json](ACTIVE_WORKFLOW.json)，公共规则是
[DUAL_AGENT_WORKFLOW_V3.md](protocols/DUAL_AGENT_WORKFLOW_V3.md)。
先读公共规则，再按用户指定角色读取 [Commander](commander/COMMANDER.md) 或
[Builder](executor/BUILDER.md) 入口。维护或咨询会话不自动认领角色。

`commander/`、`executor/` 下的现行状态是任务交接依据；顶层旧状态副本、旧 Order、
旧协议和历史证据保留用于追溯。不得用旧副本覆盖现行状态。
V3 管理后续工作权限和审核；已封存的 V2 证据仍按原始协议和哈希解释。
不要为接入 V3 修改旧证据的 schema、manifest、哈希或认证记录。
