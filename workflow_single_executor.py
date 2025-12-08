"""
LLM Agent MCM - 单工作流程执行器
执行workflow_single.md中定义的完整工作流程
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from config import (
    ProjectConfig, ModelConfig, MODELS, INTERN_MODELS, SYSTEM_PROMPTS,
    WORKFLOW_STEPS, SCORING_CRITERIA, APIProvider
)
from api_client import LLMAgent, Message, create_client
from utils import (
    read_file, write_file, file_exists, directory_exists,
    create_directory, list_files, extract_python_code,
    extract_latex_content, parse_paper_review, PaperScore,
    WorkflowState, save_state, load_state, get_timestamp,
    ProgressTracker, setup_work_directory
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """工作流程上下文"""
    work_dir: str
    model_name: str
    model_config: ModelConfig
    problem_content: str = ""
    plan_content: str = ""
    data_collected: List[str] = field(default_factory=list)
    code_files: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    paper_cn_path: str = ""
    paper_en_path: str = ""
    score: Optional[PaperScore] = None


class SingleWorkflowExecutor:
    """单工作流程执行器"""

    def __init__(
        self,
        project_config: ProjectConfig,
        model_name: str,
        work_dir: str
    ):
        self.project_config = project_config
        self.model_name = model_name
        self.work_dir = work_dir

        # 根据API提供商选择模型配置
        if project_config.provider == APIProvider.INTERN:
            self.model_config = INTERN_MODELS[model_name]
        else:
            self.model_config = MODELS[model_name]

        # 创建API客户端和Agent
        api_config = project_config.get_api_config()
        self.agent = LLMAgent(api_config, self.model_config)

        # 工作流程上下文
        self.context = WorkflowContext(
            work_dir=work_dir,
            model_name=model_name,
            model_config=self.model_config
        )

        # 状态文件路径
        self.state_file = os.path.join(work_dir, "workflow_state.json")

        # 进度跟踪
        self.progress = ProgressTracker(
            total_steps=len(WORKFLOW_STEPS),
            log_file=os.path.join(work_dir, "progress.log")
        )

    async def execute(self, resume: bool = False) -> WorkflowContext:
        """执行完整工作流程"""
        logger.info(f"开始执行工作流程: {self.model_name} @ {self.work_dir}")

        # 检查是否恢复执行
        start_step = 1
        if resume and file_exists(self.state_file):
            state = load_state(self.state_file)
            if state:
                start_step = state.current_step
                logger.info(f"从步骤 {start_step} 恢复执行")

        # 执行各步骤
        for step in WORKFLOW_STEPS:
            if step.step_id < start_step:
                continue

            self.progress.start_step(step.name)

            try:
                success = await self._execute_step(step.step_id)
                self.progress.end_step(success)

                # 保存状态
                self._save_current_state(step.step_id + 1)

            except Exception as e:
                logger.error(f"步骤 {step.step_id} ({step.name}) 执行失败: {e}")
                self.progress.end_step(False, str(e))
                self._save_current_state(step.step_id, failed=True)
                raise

        logger.info(f"工作流程执行完成: {self.model_name}")
        return self.context

    async def _execute_step(self, step_id: int) -> bool:
        """执行单个步骤"""
        step_handlers = {
            1: self._step_read_problem,
            2: self._step_search_references,
            3: self._step_design_models,
            4: self._step_collect_data,
            5: self._step_finalize_model,
            6: self._step_implement_model,
            7: self._step_create_visualizations,
            8: self._step_write_paper_cn,
            9: self._step_review_and_improve,
            10: self._step_translate_paper,
        }

        handler = step_handlers.get(step_id)
        if handler:
            return await handler()
        return False

    async def _step_read_problem(self) -> bool:
        """步骤1: 阅读题目，理解问题内容"""
        logger.info("步骤1: 阅读题目，创建plan.md")

        # 读取题目文件
        problem_dir = os.path.join(self.work_dir, "problem")
        problem_files = list_files(problem_dir, "*.md")

        if not problem_files:
            logger.error("未找到题目文件")
            return False

        problem_content = ""
        for pf in problem_files:
            content = read_file(pf)
            if content:
                problem_content += f"\n\n--- {os.path.basename(pf)} ---\n\n{content}"

        self.context.problem_content = problem_content

        # 读取workflow_single.md获取工作流程要求
        workflow_path = os.path.join(self.work_dir, "workflow_single.md")
        workflow_content = read_file(workflow_path) or ""

        # 调用LLM分析问题并创建计划
        prompt = f"""请仔细阅读以下数学建模竞赛题目，并用中文创建一份详细的解题计划。

## 题目内容
{problem_content}

## 工作流程要求
{workflow_content}

请创建plan.md文档，包含以下内容：
1. 问题背景分析
2. 核心问题识别（分解各个子问题）
3. 初步解决思路
4. 可能需要的数据类型
5. 潜在的模型方向（每个子问题列出1-3个候选模型）

请直接输出plan.md的内容，使用Markdown格式。
"""

        self.agent.set_system_prompt(SYSTEM_PROMPTS["problem_analysis"])
        plan_content = await self.agent.chat(prompt)

        # 保存plan.md
        plan_path = os.path.join(self.work_dir, "plan.md")
        write_file(plan_path, plan_content)
        self.context.plan_content = plan_content

        logger.info(f"plan.md 已创建: {plan_path}")
        return True

    async def _step_search_references(self) -> bool:
        """步骤2: 搜索相关论文和网络资料（可选）"""
        logger.info("步骤2: 搜索相关参考资料")

        # 这是可选步骤，生成搜索建议
        prompt = f"""基于以下问题和计划，请列出需要搜索的关键词和可能的数据来源。

## 问题内容
{self.context.problem_content}

## 当前计划
{self.context.plan_content}

请列出：
1. 建议搜索的学术关键词（用于论文搜索）
2. 建议搜索的通用关键词（用于网络搜索）
3. 可能的官方数据来源（政府网站、组织网站等）
4. 可能相关的学术数据库

请以Markdown格式输出搜索指南。
"""

        self.agent.clear_history()
        self.agent.set_system_prompt(SYSTEM_PROMPTS["model_design"])
        search_guide = await self.agent.chat(prompt)

        # 保存搜索指南
        guide_path = os.path.join(self.work_dir, "search_guide.md")
        write_file(guide_path, search_guide)

        logger.info("搜索指南已生成")
        return True

    async def _step_design_models(self) -> bool:
        """步骤3: 设计模型方案"""
        logger.info("步骤3: 设计模型方案")

        prompt = f"""基于以下问题和计划，请针对每个子问题设计1-3个可能的模型方案。

## 问题内容
{self.context.problem_content}

## 当前计划
{self.context.plan_content}

对于每个模型方案，请详细说明：
1. 模型名称
2. 数学原理和公式
3. 模型假设
4. 输入输出变量
5. 求解方法
6. 优点和局限性
7. 所需数据

请更新plan.md，加入详细的模型设计部分。
"""

        self.agent.clear_history()
        self.agent.set_system_prompt(SYSTEM_PROMPTS["model_design"])
        model_design = await self.agent.chat(prompt)

        # 更新plan.md
        plan_path = os.path.join(self.work_dir, "plan.md")
        current_plan = read_file(plan_path) or ""
        updated_plan = current_plan + "\n\n## 模型设计\n\n" + model_design
        write_file(plan_path, updated_plan)
        self.context.plan_content = updated_plan

        logger.info("模型设计已完成")
        return True

    async def _step_collect_data(self) -> bool:
        """步骤4: 收集数据"""
        logger.info("步骤4: 收集和整理数据")

        # 检查官方数据
        official_data_dir = os.path.join(self.work_dir, "data_official")
        search_data_dir = os.path.join(self.work_dir, "data_search")

        create_directory(search_data_dir)

        official_data_files = []
        if directory_exists(official_data_dir):
            official_data_files = list_files(official_data_dir, "*.*")

        search_data_files = list_files(search_data_dir, "*.*")

        prompt = f"""基于当前的问题和模型设计，请分析数据需求。

## 问题内容
{self.context.problem_content}

## 模型计划
{self.context.plan_content}

## 已有官方数据文件
{official_data_files}

## 已搜集数据文件
{search_data_files}

请：
1. 分析当前数据是否满足模型需求
2. 如果数据不足，列出需要补充的数据及可能的来源
3. 建议数据预处理步骤
4. 设计数据在论文中的呈现方式（表格/图表）

请创建一份数据准备报告。
"""

        self.agent.clear_history()
        self.agent.set_system_prompt(SYSTEM_PROMPTS["model_design"])
        data_report = await self.agent.chat(prompt)

        # 保存数据报告
        report_path = os.path.join(self.work_dir, "data_report.md")
        write_file(report_path, data_report)

        self.context.data_collected = official_data_files + search_data_files
        logger.info("数据分析报告已生成")
        return True

    async def _step_finalize_model(self) -> bool:
        """步骤5: 确定最终模型方案"""
        logger.info("步骤5: 确定最终模型方案")

        # 读取数据报告
        report_path = os.path.join(self.work_dir, "data_report.md")
        data_report = read_file(report_path) or ""

        prompt = f"""基于问题分析、模型设计和数据情况，请确定最终的模型方案。

## 问题内容
{self.context.problem_content}

## 模型计划
{self.context.plan_content}

## 数据情况
{data_report}

请：
1. 为每个子问题选择最合适的模型
2. 说明选择理由
3. 细化模型参数和变量定义
4. 制定详细的实现步骤

请输出最终模型方案文档。
"""

        self.agent.clear_history()
        self.agent.set_system_prompt(SYSTEM_PROMPTS["model_design"])
        final_model = await self.agent.chat(prompt)

        # 保存最终模型方案
        model_path = os.path.join(self.work_dir, "final_model.md")
        write_file(model_path, final_model)

        # 更新plan.md
        plan_path = os.path.join(self.work_dir, "plan.md")
        current_plan = read_file(plan_path) or ""
        updated_plan = current_plan + "\n\n## 最终模型方案\n\n" + final_model
        write_file(plan_path, updated_plan)

        logger.info("最终模型方案已确定")
        return True

    async def _step_implement_model(self) -> bool:
        """步骤6: 实现模型代码"""
        logger.info("步骤6: 实现模型代码")

        # 读取最终模型方案
        model_path = os.path.join(self.work_dir, "final_model.md")
        final_model = read_file(model_path) or ""

        # 创建代码目录
        code_dir = os.path.join(self.work_dir, "code")
        create_directory(code_dir)

        prompt = f"""请根据最终模型方案编写完整的Python实现代码。

## 问题内容
{self.context.problem_content}

## 最终模型方案
{final_model}

## 已有数据文件
{self.context.data_collected}

请编写：
1. 数据加载和预处理代码
2. 模型实现代码（每个子问题）
3. 模型求解代码
4. 结果输出代码

要求：
- 代码结构清晰，有适当注释
- 使用numpy, pandas, scipy等常用库
- 包含错误处理
- 输出结果保存到文件

请为每个主要模块生成独立的Python文件。输出格式：
```python:文件名.py
代码内容
```
"""

        self.agent.clear_history()
        self.agent.set_system_prompt(SYSTEM_PROMPTS["code_implementation"])
        code_response = await self.agent.chat(prompt, max_tokens=8192)

        # 解析并保存代码文件
        import re
        code_pattern = r"```python:(\S+\.py)\n(.*?)```"
        matches = re.findall(code_pattern, code_response, re.DOTALL)

        if matches:
            for filename, code in matches:
                code_path = os.path.join(code_dir, filename)
                write_file(code_path, code.strip())
                self.context.code_files.append(code_path)
                logger.info(f"代码文件已保存: {filename}")
        else:
            # 如果没有匹配到特定格式，保存为单个文件
            code = extract_python_code(code_response)
            code_path = os.path.join(code_dir, "model_implementation.py")
            write_file(code_path, code)
            self.context.code_files.append(code_path)

        logger.info(f"模型代码已生成，共 {len(self.context.code_files)} 个文件")
        return True

    async def _step_create_visualizations(self) -> bool:
        """步骤7: 创建可视化图表"""
        logger.info("步骤7: 创建可视化图表")

        # 读取配色指南
        paint_guide_path = os.path.join(self.work_dir, "PAINT_GUIDE_4.md")
        paint_guide = read_file(paint_guide_path) or ""

        # 读取最终模型方案
        model_path = os.path.join(self.work_dir, "final_model.md")
        final_model = read_file(model_path) or ""

        # 创建图片目录
        figures_dir = os.path.join(self.work_dir, "figures")
        create_directory(figures_dir)

        prompt = f"""请根据模型结果设计并编写可视化代码。

## 配色指南
{paint_guide}

## 最终模型方案
{final_model}

请设计以下可视化图表：
1. 数据分布图
2. 模型结果图
3. 对比分析图
4. 敏感性分析图
5. 其他必要的图表

要求：
- 严格遵循配色指南中的配色方案
- 图表类型丰富（柱状图、折线图、饼图、散点图、热力图、雷达图等）
- 图片中不包含标题（标题在LaTeX中定义）
- 注重视觉层次和清晰度
- 保存为高分辨率PNG格式
- 图片保存到 figures/ 目录

请输出完整的可视化代码。
```python:visualization.py
代码内容
```
"""

        self.agent.clear_history()
        self.agent.set_system_prompt(SYSTEM_PROMPTS["visualization"])
        viz_response = await self.agent.chat(prompt, max_tokens=8192)

        # 保存可视化代码
        code_dir = os.path.join(self.work_dir, "code")
        viz_code = extract_python_code(viz_response)
        viz_path = os.path.join(code_dir, "visualization.py")
        write_file(viz_path, viz_code)

        logger.info("可视化代码已生成")
        return True

    async def _step_write_paper_cn(self) -> bool:
        """步骤8: 撰写中文论文"""
        logger.info("步骤8: 撰写中文论文")

        # 读取论文模板
        paper_dir = os.path.join(self.work_dir, "paper")
        template_path = os.path.join(paper_dir, "ch01.tex")
        template = read_file(template_path) or ""

        # 读取所有准备材料
        plan_content = read_file(os.path.join(self.work_dir, "plan.md")) or ""
        final_model = read_file(os.path.join(self.work_dir, "final_model.md")) or ""
        data_report = read_file(os.path.join(self.work_dir, "data_report.md")) or ""

        # 列出所有生成的图片
        figures_dir = os.path.join(self.work_dir, "figures")
        figures = list_files(figures_dir, "*.png") if directory_exists(figures_dir) else []
        self.context.figures = figures

        prompt = f"""请根据以下材料撰写完整的MCM/ICM中文论文。

## 题目内容
{self.context.problem_content}

## 解题计划
{plan_content}

## 最终模型方案
{final_model}

## 数据情况
{data_report}

## 可用图片
{figures}

## 论文模板
{template}

## 论文要求
1. Abstract: 320-340词，第一段说明背景，后续每段说明每一问的方法和结果
2. Keywords: 3-7个
3. Background: 插入1-2张相关背景图片（左右并排）
4. Our Work: 插入流程图
5. 正文详细阐述模型建立过程
6. 行间公式使用equation环境
7. 总篇幅23-25页（不含ReportAiUse）
8. 突出模型新颖性

请直接输出完整的LaTeX论文内容。
"""

        self.agent.clear_history()
        self.agent.set_system_prompt(SYSTEM_PROMPTS["paper_writing"])
        paper_content = await self.agent.chat(prompt, max_tokens=16384)

        # 提取LaTeX内容
        latex_content = extract_latex_content(paper_content)

        # 保存论文
        paper_path = os.path.join(paper_dir, "ch01.tex")
        write_file(paper_path, latex_content)
        self.context.paper_cn_path = paper_path

        logger.info(f"中文论文已生成: {paper_path}")
        return True

    async def _step_review_and_improve(self) -> bool:
        """步骤9: 评分并改进论文"""
        logger.info("步骤9: 评审并改进论文")

        # 读取当前论文
        paper_content = read_file(self.context.paper_cn_path) or ""

        # 评审论文
        review_prompt = f"""请以严格的评审标准评审以下MCM/ICM论文。

## 题目
{self.context.problem_content}

## 论文内容
{paper_content}

## 评分标准（10分满分，4分基准）
- 模型新颖性 (25%)
- 问题理解 (15%)
- 方法论 (20%)
- 数据分析 (15%)
- 可视化 (10%)
- 写作质量 (15%)

请给出：
1. 各维度评分和总分
2. 优点列表
3. 改进方向（具体建议）
"""

        self.agent.clear_history()
        self.agent.set_system_prompt(SYSTEM_PROMPTS["paper_review"])
        review_result = await self.agent.chat(review_prompt)

        # 解析评审结果
        self.context.score = parse_paper_review(review_result)

        # 保存评审结果
        review_path = os.path.join(self.work_dir, "paper_review.md")
        write_file(review_path, review_result)

        logger.info(f"论文评审完成，总分: {self.context.score.total_score}")

        # 根据评审结果改进论文
        if self.context.score.improvements:
            improve_prompt = f"""请根据以下评审意见改进论文。

## 当前论文
{paper_content}

## 改进建议
{chr(10).join(self.context.score.improvements)}

请输出改进后的完整论文LaTeX内容。注意：直接描述改进后的内容，不要说明修改过程。
"""

            self.agent.clear_history()
            self.agent.set_system_prompt(SYSTEM_PROMPTS["paper_writing"])
            improved_paper = await self.agent.chat(improve_prompt, max_tokens=16384)

            # 保存改进后的论文
            latex_content = extract_latex_content(improved_paper)
            write_file(self.context.paper_cn_path, latex_content)

            logger.info("论文已根据评审意见改进")

        return True

    async def _step_translate_paper(self) -> bool:
        """步骤10: 翻译论文为英文"""
        logger.info("步骤10: 翻译论文为英文")

        # 读取中文论文
        cn_paper = read_file(self.context.paper_cn_path) or ""

        prompt = f"""请将以下MCM/ICM中文论文翻译为英文。

## 中文论文
{cn_paper}

要求：
1. 保持学术论文风格
2. 数学公式保持不变
3. 专业术语翻译准确
4. 语言流畅自然

请输出完整的英文LaTeX论文。
"""

        self.agent.clear_history()
        self.agent.set_system_prompt("你是一位专业的学术翻译专家，精通数学建模论文的中英翻译。")
        en_paper = await self.agent.chat(prompt, max_tokens=16384)

        # 保存英文论文
        paper_dir = os.path.join(self.work_dir, "paper")
        en_path = os.path.join(paper_dir, "en01.tex")
        latex_content = extract_latex_content(en_paper)
        write_file(en_path, latex_content)
        self.context.paper_en_path = en_path

        logger.info(f"英文论文已生成: {en_path}")
        return True

    def _save_current_state(self, next_step: int, failed: bool = False):
        """保存当前状态"""
        state = WorkflowState(
            current_step=next_step,
            completed_steps=list(range(1, next_step)),
            failed_steps=[next_step - 1] if failed else [],
            step_outputs={},
            start_time=get_timestamp(),
            last_update=get_timestamp(),
            status="failed" if failed else "running"
        )
        save_state(state, self.state_file)


async def run_single_workflow(
    project_config: ProjectConfig,
    model_name: str,
    work_dir: str,
    resume: bool = False
) -> WorkflowContext:
    """运行单个工作流程"""
    executor = SingleWorkflowExecutor(project_config, model_name, work_dir)
    return await executor.execute(resume=resume)
