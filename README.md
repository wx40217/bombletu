# bombletu

一个基于 LangGraph 与 NapCatBot 的 QQ 群聊观测/交互机器人，采用 `src` 布局组织代码。

## 目录结构

- `src/bombletu/`：核心包
  - `config.py`：环境变量读取与日志入口
  - `app.py`：Bot 应用容器、事件循环与运行入口
  - `graph.py`：LangGraph 状态机与模型配置
  - `tools.py`：供代理使用的工具集
  - `msgfmt.py`：消息解析/格式化工具
  - `cqface.py`：QQ 表情映射表
  - `__main__.py`：支持 `python -m bombletu` 启动
- `main.py`：兼容旧用法的启动脚本（内部调用 `bombletu.run`）
- `pyproject.toml`：项目与依赖配置

## 快速开始

1. 准备好 NapCatBot、OpenAI 兼容接口等依赖，并在环境中配置以下变量：`Q_USR`、`Q_NICK`、`Q_GRP`、`Q_CON`、`TZ`、`LLM_MODEL`、`LLM_API_KEY`、`LLM_BASE_URL`、`VIS_MODEL`、`VIS_API_KEY`、`VIS_BASE_URL`。
2. 安装依赖：

   ```bash
   uv sync
   ```

3. 运行机器人（任选其一）：

   ```bash
   uv run python -m bombletu
   # 或
   uv run python main.py
   ```

## 开发提示

- 需要新增模块时，请放置于 `src/bombletu/` 下并使用相对导入。
- 保持工具函数与 LangGraph 工作流的解耦，避免循环依赖。
- 统一通过 `.env` 或环境变量管理运行参数。
