"""
LLM Agent MCM - 全工作流程执行器
执行workflow_all.md中定义的完整工作流程
包括并行执行多个模型、评分比较、综合最终方案
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from config import (
    ProjectConfig, ModelConfig, MODELS, INTERN_MODELS, SYSTEM_PROMPTS,
    APIProvider, API_ENDPOINTS
)
from api_client import LLMAgent, create_client
from utils import (
    read_file, write_file, file_exists, directory_exists,
    create_directory, copy_directory, copy_file,
    list_files, parse_paper_review, PaperScore,
    get_timestamp, ProgressTracker, extract_latex_content,
    compile_latex, LaTeXCompileResult
)
from workflow_single_executor import (
    SingleWorkflowExecutor, WorkflowContext, run_single_workflow
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """模型执行结果"""
    model_name: str
    work_dir: str
    context: Optional[WorkflowContext]
    score: Optional[PaperScore]
    success: bool
    error: Optional[str] = None


@dataclass
class AllWorkflowResult:
    """全工作流程结果"""
    model_results: Dict[str, ModelResult]
    final_result: Optional[ModelResult]
    best_model: str
    synthesis_report: str


class AllWorkflowExecutor:
    """全工作流程执行器"""

    def __init__(self, project_config: ProjectConfig):
        self.project_config = project_config
        self.base_dir = project_config.base_dir

        # 源文件夹和文件
        self.source_dirs = [
            project_config.problem_dir,
            project_config.paper_dir,
            project_config.data_search_dir,
        ]
        if directory_exists(os.path.join(self.base_dir, project_config.data_official_dir)):
            self.source_dirs.append(project_config.data_official_dir)

        self.source_files = [
            project_config.paint_guide_file,
            project_config.workflow_single_file,
        ]

        # 进度跟踪
        self.progress = ProgressTracker(
            total_steps=5,  # 准备、3个并行、综合
            log_file=os.path.join(self.base_dir, "logs", "workflow_all.log")
        )

    async def execute(self) -> AllWorkflowResult:
        """执行完整工作流程"""
        logger.info("=" * 60)
        logger.info("开始执行全工作流程")
        logger.info("=" * 60)

        # 步骤1: 准备工作目录
        self.progress.start_step("准备工作目录")
        await self._prepare_directories()
        self.progress.end_step(True)

        # 步骤2: 并行执行三个模型的工作流程
        self.progress.start_step("并行执行三个模型")
        model_results = await self._run_parallel_workflows()
        self.progress.end_step(True)

        # 步骤3: 评分和比较
        self.progress.start_step("评分和比较")
        scored_results = await self._score_papers(model_results)
        self.progress.end_step(True)

        # 步骤4: 确定最佳方案
        self.progress.start_step("确定最佳方案")
        best_model = self._find_best_model(scored_results)
        self.progress.end_step(True)

        # 步骤5: 综合最终方案
        self.progress.start_step("综合最终方案")
        final_result, synthesis_report = await self._synthesize_final_solution(
            scored_results, best_model
        )
        self.progress.end_step(True)

        logger.info("=" * 60)
        logger.info("全工作流程执行完成")
        logger.info(f"最佳模型: {best_model}")
        logger.info("=" * 60)

        return AllWorkflowResult(
            model_results=scored_results,
            final_result=final_result,
            best_model=best_model,
            synthesis_report=synthesis_report
        )

    async def _prepare_directories(self):
        """准备工作目录"""
        logger.info("准备工作目录...")

        # 创建日志目录
        log_dir = os.path.join(self.base_dir, "logs")
        create_directory(log_dir)

        # 创建模板目录
        template_dir = os.path.join(self.base_dir, self.project_config.template_dir)
        if not directory_exists(template_dir):
            create_directory(template_dir)

            # 复制源文件夹
            for src_dir in self.source_dirs:
                src_path = os.path.join(self.base_dir, src_dir)
                if directory_exists(src_path):
                    dst_path = os.path.join(template_dir, src_dir)
                    copy_directory(src_path, dst_path)
                    logger.info(f"已复制目录: {src_dir}")

            # 复制源文件
            for src_file in self.source_files:
                src_path = os.path.join(self.base_dir, src_file)
                if file_exists(src_path):
                    dst_path = os.path.join(template_dir, src_file)
                    copy_file(src_path, dst_path)
                    logger.info(f"已复制文件: {src_file}")

        # 创建并行工作目录
        for model_name, work_dir_name in self.project_config.work_dirs.items():
            work_dir = os.path.join(self.base_dir, work_dir_name)
            if not directory_exists(work_dir):
                copy_directory(template_dir, work_dir)
                logger.info(f"已创建工作目录: {work_dir_name}")

        # 创建最终综合目录
        final_dir = os.path.join(self.base_dir, self.project_config.final_dir)
        if not directory_exists(final_dir):
            copy_directory(template_dir, final_dir)
            logger.info(f"已创建最终目录: {self.project_config.final_dir}")

        logger.info("工作目录准备完成")

    async def _run_parallel_workflows(self) -> Dict[str, ModelResult]:
        """并行执行三个模型的工作流程"""
        logger.info("开始并行执行三个模型的工作流程...")

        results = {}
        tasks = []

        for model_name, work_dir_name in self.project_config.work_dirs.items():
            work_dir = os.path.join(self.base_dir, work_dir_name)
            task = self._run_single_model_workflow(model_name, work_dir)
            tasks.append(task)

        # 并行执行
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for (model_name, _), result in zip(self.project_config.work_dirs.items(), completed):
            if isinstance(result, Exception):
                results[model_name] = ModelResult(
                    model_name=model_name,
                    work_dir=os.path.join(self.base_dir, self.project_config.work_dirs[model_name]),
                    context=None,
                    score=None,
                    success=False,
                    error=str(result)
                )
                logger.error(f"模型 {model_name} 执行失败: {result}")
            else:
                results[model_name] = result
                logger.info(f"模型 {model_name} 执行完成")

        return results

    async def _run_single_model_workflow(
        self,
        model_name: str,
        work_dir: str
    ) -> ModelResult:
        """执行单个模型的工作流程"""
        logger.info(f"开始执行模型 {model_name} 的工作流程...")

        try:
            context = await run_single_workflow(
                self.project_config,
                model_name,
                work_dir
            )

            return ModelResult(
                model_name=model_name,
                work_dir=work_dir,
                context=context,
                score=context.score,
                success=True
            )

        except Exception as e:
            logger.error(f"模型 {model_name} 工作流程失败: {e}")
            return ModelResult(
                model_name=model_name,
                work_dir=work_dir,
                context=None,
                score=None,
                success=False,
                error=str(e)
            )

    async def _score_papers(
        self,
        model_results: Dict[str, ModelResult]
    ) -> Dict[str, ModelResult]:
        """对所有论文进行评分"""
        logger.info("开始对论文进行评分比较...")

        # 根据API提供商选择模型配置
        api_config = self.project_config.get_api_config()
        if self.project_config.provider == APIProvider.INTERN:
            scoring_model = INTERN_MODELS["claude"]
        else:
            scoring_model = MODELS["claude"]

        scoring_agent = LLMAgent(api_config, scoring_model)
        scoring_agent.set_system_prompt(SYSTEM_PROMPTS["paper_review"])

        for model_name, result in model_results.items():
            if not result.success:
                continue

            # 读取论文内容
            paper_dir = os.path.join(result.work_dir, "paper")
            paper_path = os.path.join(paper_dir, "ch01.tex")
            paper_content = read_file(paper_path)

            if not paper_content:
                continue

            # 读取问题内容
            problem_dir = os.path.join(result.work_dir, "problem")
            problem_files = list_files(problem_dir, "*.md")
            problem_content = ""
            for pf in problem_files:
                content = read_file(pf)
                if content:
                    problem_content += content + "\n"

            # 评分
            prompt = f"""请对以下MCM/ICM论文进行严格评分。

## 题目
{problem_content}

## 论文内容（{model_name}模型生成）
{paper_content}

## 评分标准（10分满分，4分基准分）
- 模型新颖性 (25%): 模型是否具有创新性和独特性
- 问题理解 (15%): 对问题的理解是否深入准确
- 方法论 (20%): 解决方法是否合理严谨
- 数据分析 (15%): 数据处理和分析是否充分
- 可视化 (10%): 图表是否美观清晰
- 写作质量 (15%): 论文结构和表达是否清晰

请按以下格式输出：
1. 各维度评分
2. 总分
3. 优点（列表）
4. 改进方向（列表）
"""

            try:
                review_result = await scoring_agent.single_query(
                    SYSTEM_PROMPTS["paper_review"],
                    prompt
                )

                score = parse_paper_review(review_result)
                result.score = score

                # 保存评分结果
                review_path = os.path.join(result.work_dir, "unified_review.md")
                write_file(review_path, review_result)

                logger.info(f"模型 {model_name} 评分: {score.total_score}")

            except Exception as e:
                logger.error(f"评分模型 {model_name} 失败: {e}")

        return model_results

    def _find_best_model(self, model_results: Dict[str, ModelResult]) -> str:
        """找出最佳模型"""
        best_model = None
        best_score = -1

        for model_name, result in model_results.items():
            if result.success and result.score:
                if result.score.total_score > best_score:
                    best_score = result.score.total_score
                    best_model = model_name

        logger.info(f"最佳模型: {best_model} (得分: {best_score})")
        return best_model or "claude"

    async def _synthesize_final_solution(
        self,
        model_results: Dict[str, ModelResult],
        best_model: str
    ) -> tuple[ModelResult, str]:
        """综合最终解决方案"""
        logger.info("开始综合最终解决方案...")

        final_dir = os.path.join(self.base_dir, self.project_config.final_dir)

        # 收集所有论文和评分
        papers = {}
        reviews = {}
        for model_name, result in model_results.items():
            if result.success:
                paper_path = os.path.join(result.work_dir, "paper", "ch01.tex")
                papers[model_name] = read_file(paper_path) or ""

                review_path = os.path.join(result.work_dir, "unified_review.md")
                reviews[model_name] = read_file(review_path) or ""

        # 读取问题内容
        problem_dir = os.path.join(final_dir, "problem")
        problem_files = list_files(problem_dir, "*.md")
        problem_content = ""
        for pf in problem_files:
            content = read_file(pf)
            if content:
                problem_content += content + "\n"

        # 使用Claude进行综合（根据API提供商选择模型配置）
        api_config = self.project_config.get_api_config()
        if self.project_config.provider == APIProvider.INTERN:
            synthesis_model = INTERN_MODELS["claude"]
        else:
            synthesis_model = MODELS["claude"]

        synthesis_agent = LLMAgent(api_config, synthesis_model)

        # 生成综合报告
        report_prompt = f"""请分析以下三个模型生成的MCM/ICM论文，综合各方案的优点，形成最终的综合方案。

## 题目
{problem_content}

## Claude模型方案
评分: {model_results.get('claude', ModelResult('', '', None, None, False)).score.total_score if model_results.get('claude') and model_results['claude'].score else 'N/A'}
{reviews.get('claude', '无')}

## Gemini模型方案
评分: {model_results.get('gemini', ModelResult('', '', None, None, False)).score.total_score if model_results.get('gemini') and model_results['gemini'].score else 'N/A'}
{reviews.get('gemini', '无')}

## GPT模型方案
评分: {model_results.get('gpt', ModelResult('', '', None, None, False)).score.total_score if model_results.get('gpt') and model_results['gpt'].score else 'N/A'}
{reviews.get('gpt', '无')}

请输出：
1. 各方案对比分析
2. 各方案的独特亮点
3. 综合方案的设计思路
4. 最终方案的关键创新点
"""

        synthesis_report = await synthesis_agent.single_query(
            SYSTEM_PROMPTS["paper_synthesis"],
            report_prompt
        )

        # 保存综合报告
        report_path = os.path.join(final_dir, "synthesis_report.md")
        write_file(report_path, synthesis_report)

        # 生成最终论文
        final_paper_prompt = f"""请基于以下三个模型的论文，综合各方案的优点，生成最终的MCM/ICM论文。

## 题目
{problem_content}

## 综合报告
{synthesis_report}

## Claude模型论文
{papers.get('claude', '')}

## Gemini模型论文
{papers.get('gemini', '')}

## GPT模型论文
{papers.get('gpt', '')}

请综合以上方案的优点，生成一篇完整的、高质量的MCM/ICM论文。
要求：
1. 取各方案之长
2. 模型要具有创新性
3. 论述要严谨完整
4. 直接输出完整的LaTeX内容

注意：直接描述最终方案，不要说明综合过程或比较不同版本。
"""

        final_paper = await synthesis_agent.single_query(
            SYSTEM_PROMPTS["paper_writing"],
            final_paper_prompt,
            max_tokens=16384
        )

        # 保存最终论文
        paper_dir = os.path.join(final_dir, "paper")
        final_paper_path = os.path.join(paper_dir, "ch01.tex")
        latex_content = extract_latex_content(final_paper)
        write_file(final_paper_path, latex_content)

        # 翻译为英文
        translate_prompt = f"""请将以下MCM/ICM中文论文翻译为英文。

{latex_content}

要求：
1. 保持学术论文风格
2. 数学公式保持不变
3. 专业术语翻译准确
4. 语言流畅自然

请输出完整的英文LaTeX论文。
"""

        en_paper = await synthesis_agent.single_query(
            "你是一位专业的学术翻译专家，精通数学建模论文的中英翻译。",
            translate_prompt,
            max_tokens=16384
        )

        en_paper_path = os.path.join(paper_dir, "en01.tex")
        en_latex = extract_latex_content(en_paper)
        write_file(en_paper_path, en_latex)

        # 编译LaTeX论文生成PDF
        logger.info("编译最终论文...")
        compile_results = []

        # 编译中文论文
        cn_result = compile_latex(final_paper_path)
        cn_pdf_path = None
        if cn_result.success:
            cn_pdf_path = cn_result.pdf_path
            logger.info(f"中文论文PDF生成成功: {cn_pdf_path}")
            compile_results.append(("中文论文", True, cn_pdf_path))
        else:
            logger.warning(f"中文论文编译失败: {cn_result.error_message}")
            compile_results.append(("中文论文", False, cn_result.error_message))
            # 保存编译日志
            log_path = os.path.join(paper_dir, "ch01_compile.log")
            write_file(log_path, cn_result.log_output)

        # 编译英文论文
        en_result = compile_latex(en_paper_path)
        en_pdf_path = None
        if en_result.success:
            en_pdf_path = en_result.pdf_path
            logger.info(f"英文论文PDF生成成功: {en_pdf_path}")
            compile_results.append(("英文论文", True, en_pdf_path))
        else:
            logger.warning(f"英文论文编译失败: {en_result.error_message}")
            compile_results.append(("英文论文", False, en_result.error_message))
            log_path = os.path.join(paper_dir, "en01_compile.log")
            write_file(log_path, en_result.log_output)

        # 生成编译报告
        report_lines = ["# 最终论文 LaTeX 编译报告\n"]
        for name, success, info in compile_results:
            status = "✓ 成功" if success else "✗ 失败"
            report_lines.append(f"## {name}: {status}")
            report_lines.append(f"- {info}\n")

        compile_report_path = os.path.join(final_dir, "compile_report.md")
        write_file(compile_report_path, '\n'.join(report_lines))

        # 创建最终结果
        final_result = ModelResult(
            model_name="synthesis",
            work_dir=final_dir,
            context=None,
            score=None,
            success=True
        )

        logger.info(f"最终论文已生成: {final_paper_path}")
        logger.info(f"英文版论文已生成: {en_paper_path}")
        if cn_pdf_path:
            logger.info(f"中文PDF: {cn_pdf_path}")
        if en_pdf_path:
            logger.info(f"英文PDF: {en_pdf_path}")

        return final_result, synthesis_report


async def run_all_workflow(project_config: ProjectConfig) -> AllWorkflowResult:
    """运行完整工作流程"""
    executor = AllWorkflowExecutor(project_config)
    return await executor.execute()
