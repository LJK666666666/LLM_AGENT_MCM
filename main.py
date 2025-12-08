"""
LLM Agent MCM - 主程序入口
美国大学生数学建模竞赛自动化Agent系统

使用方法:
    python main.py                    # 执行完整工作流程
    python main.py --mode single      # 仅执行单个模型工作流程
    python main.py --mode parallel    # 并行执行三个模型
    python main.py --mode synthesis   # 仅执行综合步骤
    python main.py --resume           # 从断点恢复执行
    python main.py --intern           # 使用Intern API（书生·浦语）
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime

from config import ProjectConfig, APIProvider, MODELS, INTERN_MODELS
from utils import (
    setup_logging, read_file, write_file, create_directory,
    directory_exists, file_exists
)
from workflow_single_executor import run_single_workflow
from workflow_all_executor import run_all_workflow, AllWorkflowExecutor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LLM Agent MCM - 美国大学生数学建模竞赛自动化Agent系统"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "single", "parallel", "synthesis"],
        default="all",
        help="执行模式: all(完整流程), single(单模型), parallel(并行), synthesis(仅综合)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["claude", "gemini", "gpt"],
        default="claude",
        help="单模型模式下使用的模型"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="从断点恢复执行"
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="工作目录路径"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API密钥（优先使用，否则从配置文件读取）"
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "openai", "anthropic", "intern"],
        default="openrouter",
        help="API提供商"
    )

    parser.add_argument(
        "--intern",
        action="store_true",
        help="使用Intern API（书生·浦语 Intern-S1），启用后所有流程均使用Intern API"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )

    return parser.parse_args()


def load_api_key(base_dir: str, use_intern: bool = False) -> str:
    """从配置文件加载API密钥

    Args:
        base_dir: 基础目录
        use_intern: 是否使用Intern API

    Returns:
        API密钥字符串
    """
    # 如果使用Intern API，优先从intern_api.md读取
    if use_intern:
        api_file = os.path.join(base_dir, "intern_api.md")
        if file_exists(api_file):
            content = read_file(api_file)
            if content:
                lines = content.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    # Intern API key 以 sk- 开头
                    if line.startswith("sk-") or line.startswith("eyJ"):
                        return line

        # 尝试从环境变量读取
        env_key = os.environ.get("INTERN_API_KEY")
        if env_key:
            return env_key

    # 尝试从openrouter_api.md读取
    api_file = os.path.join(base_dir, "openrouter_api.md")
    if file_exists(api_file):
        content = read_file(api_file)
        if content:
            # 提取API key（假设第一行是key）
            lines = content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("sk-"):
                    return line

    # 尝试从API_key.md读取
    api_file = os.path.join(base_dir, "API_key.md")
    if file_exists(api_file):
        content = read_file(api_file)
        if content:
            lines = content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("sk-"):
                    return line

    # 尝试从环境变量读取
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key

    return ""


def create_project_config(args, base_dir: str) -> ProjectConfig:
    """创建项目配置"""
    # 检查是否使用Intern API
    use_intern = args.intern or args.provider == "intern"

    # 获取API密钥
    api_key = args.api_key or load_api_key(base_dir, use_intern=use_intern)
    if not api_key:
        if use_intern:
            raise ValueError("未找到Intern API密钥，请通过--api-key参数提供或在intern_api.md中设置")
        else:
            raise ValueError("未找到API密钥，请通过--api-key参数提供或在配置文件中设置")

    # 确定API提供商
    if use_intern:
        provider = APIProvider.INTERN
    else:
        provider_map = {
            "openrouter": APIProvider.OPENROUTER,
            "openai": APIProvider.OPENAI,
            "anthropic": APIProvider.ANTHROPIC,
            "intern": APIProvider.INTERN,
        }
        provider = provider_map.get(args.provider, APIProvider.OPENROUTER)

    return ProjectConfig(
        base_dir=base_dir,
        api_key=api_key,
        provider=provider
    )


async def run_single_mode(config: ProjectConfig, model_name: str, resume: bool):
    """运行单模型模式"""
    work_dir = os.path.join(config.base_dir, config.work_dirs[model_name])

    # 如果工作目录不存在，从模板创建
    if not directory_exists(work_dir):
        template_dir = os.path.join(config.base_dir, config.template_dir)
        if directory_exists(template_dir):
            from utils import copy_directory
            copy_directory(template_dir, work_dir)
        else:
            # 直接创建并复制源文件
            from workflow_all_executor import AllWorkflowExecutor
            executor = AllWorkflowExecutor(config)
            await executor._prepare_directories()

    context = await run_single_workflow(config, model_name, work_dir, resume)

    print("\n" + "=" * 60)
    print(f"单模型工作流程完成: {model_name}")
    print(f"工作目录: {work_dir}")
    if context.score:
        print(f"论文评分: {context.score.total_score}")
    print("=" * 60)

    return context


async def run_parallel_mode(config: ProjectConfig):
    """运行并行模式"""
    from workflow_all_executor import AllWorkflowExecutor

    executor = AllWorkflowExecutor(config)

    # 准备目录
    await executor._prepare_directories()

    # 并行执行
    results = await executor._run_parallel_workflows()

    print("\n" + "=" * 60)
    print("并行工作流程完成")
    for model_name, result in results.items():
        status = "成功" if result.success else f"失败: {result.error}"
        score = result.score.total_score if result.score else "N/A"
        print(f"  {model_name}: {status}, 评分: {score}")
    print("=" * 60)

    return results


async def run_synthesis_mode(config: ProjectConfig):
    """运行综合模式"""
    from workflow_all_executor import AllWorkflowExecutor

    executor = AllWorkflowExecutor(config)

    # 检查并行执行结果是否存在
    model_results = {}
    for model_name, work_dir_name in config.work_dirs.items():
        work_dir = os.path.join(config.base_dir, work_dir_name)
        paper_path = os.path.join(work_dir, "paper", "ch01.tex")

        if file_exists(paper_path):
            from workflow_all_executor import ModelResult
            model_results[model_name] = ModelResult(
                model_name=model_name,
                work_dir=work_dir,
                context=None,
                score=None,
                success=True
            )

    if not model_results:
        print("错误: 未找到已完成的模型工作流程，请先运行parallel模式")
        return None

    # 评分
    scored_results = await executor._score_papers(model_results)

    # 找最佳模型
    best_model = executor._find_best_model(scored_results)

    # 综合
    final_result, synthesis_report = await executor._synthesize_final_solution(
        scored_results, best_model
    )

    print("\n" + "=" * 60)
    print("综合工作流程完成")
    print(f"最佳模型: {best_model}")
    print(f"最终论文: {os.path.join(config.base_dir, config.final_dir, 'paper', 'ch01.tex')}")
    print("=" * 60)

    return final_result


async def run_all_mode(config: ProjectConfig):
    """运行完整模式"""
    result = await run_all_workflow(config)

    print("\n" + "=" * 60)
    print("完整工作流程执行完成")
    print("-" * 60)
    print("各模型评分:")
    for model_name, model_result in result.model_results.items():
        status = "成功" if model_result.success else f"失败"
        score = model_result.score.total_score if model_result.score else "N/A"
        print(f"  {model_name}: {status}, 评分: {score}")
    print("-" * 60)
    print(f"最佳模型: {result.best_model}")
    print(f"最终论文目录: {os.path.join(config.base_dir, config.final_dir)}")
    print("=" * 60)

    return result


def print_banner():
    """打印启动横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     LLM Agent MCM - 数学建模竞赛自动化Agent系统              ║
║                                                               ║
║     支持模型: Claude, Gemini, GPT                            ║
║     API: OpenRouter / Intern (书生·浦语)                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main():
    """主函数"""
    print_banner()

    # 解析参数
    args = parse_args()

    # 确定基础目录
    base_dir = args.work_dir or os.path.dirname(os.path.abspath(__file__))

    # 设置日志
    log_dir = os.path.join(base_dir, "logs")
    create_directory(log_dir)
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_dir, log_level)

    logger.info(f"启动LLM Agent MCM")
    logger.info(f"执行模式: {args.mode}")
    logger.info(f"基础目录: {base_dir}")

    try:
        # 创建项目配置
        config = create_project_config(args, base_dir)
        logger.info(f"API提供商: {config.provider.value}")

        # 根据模式执行
        if args.mode == "single":
            await run_single_mode(config, args.model, args.resume)

        elif args.mode == "parallel":
            await run_parallel_mode(config)

        elif args.mode == "synthesis":
            await run_synthesis_mode(config)

        else:  # all
            await run_all_mode(config)

        logger.info("执行完成")

    except KeyboardInterrupt:
        logger.info("用户中断执行")
        print("\n执行已中断")
        sys.exit(1)

    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
