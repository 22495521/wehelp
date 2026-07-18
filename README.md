# wehelp

## uv 操作說明

本專案使用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本與套件依賴。


### 安裝依賴

複製本專案後，於專案根目錄執行：

```bash
uv sync
```

這會依照 `pyproject.toml` 與 `uv.lock` 建立虛擬環境（`.venv`）並安裝所有依賴，確保版本與鎖定檔一致。

### 執行程式

不需手動啟用虛擬環境，直接用 `uv run` 執行：

```bash
uv run python week6/xxx.py
```

### 新增 / 移除套件

```bash
# 新增套件
uv add <package_name>

# 新增指定版本
uv add "<package_name>>=1.0.0"

# 移除套件
uv remove <package_name>
```

新增或移除套件後，`pyproject.toml` 與 `uv.lock` 會自動更新，記得一併提交至版本控制。

### 更新依賴

```bash
# 更新所有套件到符合版本限制的最新版
uv lock --upgrade

# 更新單一套件
uv lock --upgrade-package <package_name>
```

### 常用指令

| 指令 | 說明 |
| --- | --- |
| `uv sync` | 依照 `uv.lock` 安裝/同步環境 |
| `uv run <cmd>` | 在虛擬環境中執行指令 |
| `uv add <pkg>` | 新增套件依賴 |
| `uv remove <pkg>` | 移除套件依賴 |
| `uv lock` | 產生/更新 `uv.lock` |
| `uv python list` | 列出可用的 Python 版本 |
| `uv python install 3.12` | 安裝指定版本的 Python |
