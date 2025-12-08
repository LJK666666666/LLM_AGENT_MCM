"""
LLM Agent MCM - 配置文件
美国大学生数学建模竞赛自动化Agent系统
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class APIProvider(Enum):
    """API提供商枚举"""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    INTERN = "intern"  # 书生·浦语 Intern-S1


@dataclass
class ModelConfig:
    """模型配置"""
    model_id: str
    provider: APIProvider
    max_tokens: int = 8192
    temperature: float = 0.7


@dataclass
class APIConfig:
    """API配置"""
    provider: APIProvider
    api_key: str
    base_url: str


# 预定义模型配置
MODELS = {
    "claude": ModelConfig(
        model_id="anthropic/claude-sonnet-4",  # 使用实际可用的模型
        provider=APIProvider.OPENROUTER,
        max_tokens=8192,
        temperature=0.7
    ),
    "gemini": ModelConfig(
        model_id="google/gemini-2.0-flash-001",  # 使用实际可用的模型
        provider=APIProvider.OPENROUTER,
        max_tokens=8192,
        temperature=0.7
    ),
    "gpt": ModelConfig(
        model_id="openai/gpt-4o",  # 使用实际可用的模型
        provider=APIProvider.OPENROUTER,
        max_tokens=8192,
        temperature=0.7
    ),
}

# Intern API模型配置（书生·浦语）
INTERN_MODELS = {
    "claude": ModelConfig(
        model_id="intern-s1",
        provider=APIProvider.INTERN,
        max_tokens=8192,
        temperature=0.7
    ),
    "gemini": ModelConfig(
        model_id="intern-s1",
        provider=APIProvider.INTERN,
        max_tokens=8192,
        temperature=0.7
    ),
    "gpt": ModelConfig(
        model_id="intern-s1",
        provider=APIProvider.INTERN,
        max_tokens=8192,
        temperature=0.7
    ),
}


# API端点配置
API_ENDPOINTS = {
    APIProvider.OPENROUTER: "https://openrouter.ai/api/v1/chat/completions",
    APIProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
    APIProvider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
    APIProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta/models",
    APIProvider.INTERN: "https://chat.intern-ai.org.cn/api/v1/chat/completions",
}


@dataclass
class ProjectConfig:
    """项目配置"""
    base_dir: str
    api_key: str
    provider: APIProvider = APIProvider.OPENROUTER

    # 源文件夹
    problem_dir: str = "problem"
    paper_dir: str = "paper"
    data_search_dir: str = "data_search"
    data_official_dir: str = "data_official"

    # 工作流程文件
    paint_guide_file: str = "PAINT_GUIDE_4.md"
    workflow_single_file: str = "workflow_single.md"

    # 模板和工作文件夹
    template_dir: str = "program_template"

    # 并行工作文件夹
    work_dirs: Dict[str, str] = field(default_factory=lambda: {
        "claude": "program_01claude",
        "gemini": "program_02gemini",
        "gpt": "program_03gpt",
    })

    # 最终综合文件夹
    final_dir: str = "program_11claude"

    def get_full_path(self, relative_path: str) -> str:
        """获取完整路径"""
        return os.path.join(self.base_dir, relative_path)

    def get_api_config(self) -> APIConfig:
        """获取API配置"""
        return APIConfig(
            provider=self.provider,
            api_key=self.api_key,
            base_url=API_ENDPOINTS[self.provider]
        )


@dataclass
class WorkflowStep:
    """工作流程步骤"""
    step_id: int
    name: str
    description: str
    is_optional: bool = False
    requires_user_input: bool = False


# 单工作流程步骤定义
WORKFLOW_STEPS = [
    WorkflowStep(1, "read_problem", "阅读题目，理解问题内容，创建plan.md"),
    WorkflowStep(2, "search_references", "搜索相关论文和网络资料", is_optional=True),
    WorkflowStep(3, "design_models", "针对每个部分构思模型方案"),
    WorkflowStep(4, "collect_data", "搜索和收集相关数据"),
    WorkflowStep(5, "finalize_model", "结合数据确定最终模型方案"),
    WorkflowStep(6, "implement_model", "编写代码实现模型求解"),
    WorkflowStep(7, "create_visualizations", "编写代码绘制可视化图片"),
    WorkflowStep(8, "write_paper_cn", "用中文撰写论文ch01.tex"),
    WorkflowStep(9, "review_and_improve", "评分并改进论文", requires_user_input=True),
    WorkflowStep(10, "translate_paper", "翻译论文为英文版en01.tex", requires_user_input=True),
]


# 论文评分标准
SCORING_CRITERIA = {
    "model_novelty": {
        "name": "模型新颖性",
        "weight": 0.25,
        "description": "模型是否具有创新性和独特性"
    },
    "problem_understanding": {
        "name": "问题理解",
        "weight": 0.15,
        "description": "对问题的理解是否深入准确"
    },
    "methodology": {
        "name": "方法论",
        "weight": 0.20,
        "description": "解决方法是否合理严谨"
    },
    "data_analysis": {
        "name": "数据分析",
        "weight": 0.15,
        "description": "数据处理和分析是否充分"
    },
    "visualization": {
        "name": "可视化",
        "weight": 0.10,
        "description": "图表是否美观清晰"
    },
    "writing_quality": {
        "name": "写作质量",
        "weight": 0.15,
        "description": "论文结构和表达是否清晰"
    },
}


# 系统提示词模板
SYSTEM_PROMPTS = {
    "problem_analysis": """你是一位经验丰富的数学建模专家，专门指导美国大学生数学建模竞赛(MCM/ICM)。
请仔细阅读题目，理解问题内容，并用中文创建一份详细的解题计划。

你的输出应该包括：
1. 问题背景分析
2. 核心问题识别
3. 初步解决思路
4. 可能需要的数据
5. 潜在的模型方向
""",

    "model_design": """你是一位数学建模专家，请针对给定的问题设计1-3个可能的模型方案。

对于每个方案，请说明：
1. 模型名称和基本思路
2. 模型的数学表述
3. 模型的优缺点
4. 所需数据
""",

    "code_implementation": """你是一位精通Python的数据科学家，请编写代码实现给定的数学模型。

要求：
1. 代码清晰、有良好的注释
2. 使用常见的科学计算库(numpy, scipy, pandas等)
3. 包含数据处理、模型求解、结果输出
""",

    "visualization": """你是一位数据可视化专家，请根据PAINT_GUIDE_4.md中的配色指南绘制美观的图表。

要求：
1. 严格遵循配色指南
2. 图表形式丰富多样
3. 图片中不能有标题（标题在latex中定义）
4. 注重视觉层次和清晰度
""",

    "paper_writing": """你是一位学术论文写作专家，请按照MCM/ICM论文规范撰写论文。

要求：
1. Abstract: 320-340词
2. Keywords: 3-7个
3. 突出模型新颖性
4. 行间公式使用equation环境
5. 总篇幅23-25页(不含ReportAiUse)
""",

    "paper_review": """你是一位严格的论文评审专家，请以10分为满分、4分为基准分对论文进行评分。

评分维度：
1. 模型新颖性 (25%)
2. 问题理解 (15%)
3. 方法论 (20%)
4. 数据分析 (15%)
5. 可视化 (10%)
6. 写作质量 (15%)

请给出：
1. 各维度得分和总分
2. 优点列表
3. 改进方向
""",

    "paper_synthesis": """你是一位资深的数学建模专家，请综合分析多份解决方案，取长补短，形成最优的最终方案。

请：
1. 对比各方案的优缺点
2. 提取各方案的亮点
3. 综合形成最终方案
4. 确保最终方案的完整性和一致性
""",
}
