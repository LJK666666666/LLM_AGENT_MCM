# LLM Agent MCM

本项目为美国大学生数学建模竞赛 (MCM/ICM) 自动化Agent系统。同时也是书生大模型实战营项目。

![技术架构图](architechture3.png "技术架构图")

## 功能特性

- **多模型并行**: 支持同时使用 Claude、Gemini、GPT 等多个模型并行生成解决方案
- **完整工作流程**: 从问题分析、模型设计、代码实现到论文撰写的全流程自动化
- **智能评分**: 对生成的论文进行自动评分和改进建议
- **方案综合**: 综合多个模型的优点，生成最终的高质量论文
- **API兼容**: 支持 OpenRouter API 、Intern API (--intern)，并预留 OpenAI、Anthropic 等 API 接口

## 系统架构

```
LLM_agent_MCM/
├── config.py                    # 配置文件（模型、API、工作流程）
├── api_client.py                # API客户端（支持多种API）
├── utils.py                     # 工具函数
├── workflow_single_executor.py  # 单工作流程执行器
├── workflow_all_executor.py     # 全工作流程执行器
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖包
├── problem/                     # 题目文件夹
├── paper/                       # 论文模板
├── data_official/               # 官方数据（如有）
├── data_search/                 # 搜集的数据
├── PAINT_GUIDE_4.md            # 配色指南
└── workflow_single.md          # 单工作流程说明
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 完整工作流程（推荐）

```bash
python main.py
```

这将执行完整的工作流程：
1. 准备工作目录
2. 并行执行三个模型的工作流程
3. 对论文进行评分比较
4. 综合最终解决方案

### 单模型模式

```bash
# 使用Claude模型
python main.py --mode single --model claude

# 使用Gemini模型
python main.py --mode single --model gemini

# 使用GPT模型
python main.py --mode single --model gpt
```

### 并行模式（仅执行三个模型）

```bash
python main.py --mode parallel
```

### 综合模式（仅执行综合步骤）

```bash
python main.py --mode synthesis
```

### 从断点恢复

```bash
python main.py --resume
```

### 指定API密钥

```bash
python main.py --api-key sk-your-api-key
```

## 配置

### API密钥

API密钥可以通过以下方式配置（按优先级）：
1. 命令行参数 `--api-key`
2. 配置文件 `openrouter_api.md` 或 `API_key.md`
3. 环境变量 `OPENROUTER_API_KEY`

### 模型配置

在 `config.py` 中可以修改模型配置：

```python
MODELS = {
    "claude": ModelConfig(
        model_id="anthropic/claude-sonnet-4",
        provider=APIProvider.OPENROUTER,
        max_tokens=8192,
        temperature=0.7
    ),
    # ...
}
```

## 工作流程说明

### 单工作流程（10步）

1. **阅读题目**: 分析问题，创建解题计划 `plan.md`
2. **搜索资料**: 生成搜索指南（可选）
3. **设计模型**: 为每个子问题设计模型方案
4. **收集数据**: 分析数据需求，整理已有数据
5. **确定方案**: 选择最终模型方案
6. **实现代码**: 编写Python代码实现模型
7. **可视化**: 创建图表（遵循配色指南）
8. **撰写论文**: 生成中文论文 `ch01.tex`
9. **评审改进**: 评分并改进论文
10. **翻译论文**: 翻译为英文 `en01.tex`

### 全工作流程

1. **准备目录**: 创建模板和工作目录
2. **并行执行**: 三个模型同时执行单工作流程
3. **统一评分**: 对所有论文进行评分比较
4. **确定最佳**: 选出最佳模型
5. **综合方案**: 综合各方案优点，生成最终论文

## 输出结构

执行完成后，将生成以下目录结构：

```
LLM_agent_MCM/
├── program_template/    # 模板目录
├── program_01claude/    # Claude模型工作目录
├── program_02gemini/    # Gemini模型工作目录
├── program_03gpt/       # GPT模型工作目录
├── program_11claude/    # 最终综合目录
│   ├── paper/
│   │   ├── ch01.tex    # 最终中文论文
│   │   └── en01.tex    # 最终英文论文
│   └── synthesis_report.md  # 综合报告
└── logs/               # 日志目录
```

## 注意事项

1. 确保有足够的API额度，完整流程需要大量API调用
2. 建议先使用单模型模式测试
3. 数据搜索步骤可能需要人工干预
4. 生成的代码和论文需要人工审核
