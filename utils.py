"""
LLM Agent MCM - 工具函数模块
提供文件操作、目录管理、日志等工具函数
"""

import os
import shutil
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict


def setup_logging(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mcm_agent_{timestamp}.log")

    # 创建logger
    logger = logging.getLogger("mcm_agent")
    logger.setLevel(log_level)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def copy_directory(src: str, dst: str, ignore_patterns: List[str] = None) -> bool:
    """复制目录"""
    try:
        if ignore_patterns:
            ignore = shutil.ignore_patterns(*ignore_patterns)
            shutil.copytree(src, dst, ignore=ignore)
        else:
            shutil.copytree(src, dst)
        return True
    except Exception as e:
        logging.error(f"Failed to copy directory {src} to {dst}: {e}")
        return False


def copy_file(src: str, dst: str) -> bool:
    """复制文件"""
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logging.error(f"Failed to copy file {src} to {dst}: {e}")
        return False


def create_directory(path: str) -> bool:
    """创建目录"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}")
        return False


def read_file(path: str, encoding: str = "utf-8") -> Optional[str]:
    """读取文件内容"""
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read file {path}: {e}")
        return None


def write_file(path: str, content: str, encoding: str = "utf-8") -> bool:
    """写入文件内容"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logging.error(f"Failed to write file {path}: {e}")
        return False


def append_file(path: str, content: str, encoding: str = "utf-8") -> bool:
    """追加文件内容"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logging.error(f"Failed to append to file {path}: {e}")
        return False


def list_files(directory: str, pattern: str = "*") -> List[str]:
    """列出目录中的文件"""
    try:
        path = Path(directory)
        return [str(f) for f in path.glob(pattern)]
    except Exception as e:
        logging.error(f"Failed to list files in {directory}: {e}")
        return []


def file_exists(path: str) -> bool:
    """检查文件是否存在"""
    return os.path.isfile(path)


def directory_exists(path: str) -> bool:
    """检查目录是否存在"""
    return os.path.isdir(path)


@dataclass
class WorkflowState:
    """工作流程状态"""
    current_step: int
    completed_steps: List[int]
    failed_steps: List[int]
    step_outputs: Dict[int, Any]
    start_time: str
    last_update: str
    status: str  # "running", "paused", "completed", "failed"


def save_state(state: WorkflowState, path: str) -> bool:
    """保存工作流程状态"""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"Failed to save state to {path}: {e}")
        return False


def load_state(path: str) -> Optional[WorkflowState]:
    """加载工作流程状态"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return WorkflowState(**data)
    except Exception as e:
        logging.error(f"Failed to load state from {path}: {e}")
        return None


def extract_code_blocks(text: str, language: str = None) -> List[str]:
    """从文本中提取代码块"""
    if language:
        pattern = rf"```{language}\n(.*?)```"
    else:
        pattern = r"```(?:\w+)?\n(.*?)```"

    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def extract_latex_content(text: str) -> str:
    """从文本中提取LaTeX内容"""
    # 尝试提取latex代码块
    latex_blocks = extract_code_blocks(text, "latex")
    if latex_blocks:
        return "\n\n".join(latex_blocks)

    # 尝试提取tex代码块
    tex_blocks = extract_code_blocks(text, "tex")
    if tex_blocks:
        return "\n\n".join(tex_blocks)

    # 如果没有代码块，返回原文本
    return text


def extract_python_code(text: str) -> str:
    """从文本中提取Python代码"""
    python_blocks = extract_code_blocks(text, "python")
    if python_blocks:
        return "\n\n".join(python_blocks)
    return text


def parse_score(text: str) -> Optional[float]:
    """从文本中解析分数"""
    # 匹配各种分数格式
    patterns = [
        r"总分[：:]\s*(\d+(?:\.\d+)?)",
        r"总评[：:]\s*(\d+(?:\.\d+)?)",
        r"评分[：:]\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*[/／]\s*10",
        r"(\d+(?:\.\d+)?)\s*分",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    return None


def parse_scoring_breakdown(text: str) -> Dict[str, float]:
    """解析分项评分"""
    scores = {}

    patterns = {
        "model_novelty": r"(?:模型新颖性|创新性)[：:]\s*(\d+(?:\.\d+)?)",
        "problem_understanding": r"(?:问题理解|理解程度)[：:]\s*(\d+(?:\.\d+)?)",
        "methodology": r"(?:方法论|方法)[：:]\s*(\d+(?:\.\d+)?)",
        "data_analysis": r"(?:数据分析|数据处理)[：:]\s*(\d+(?:\.\d+)?)",
        "visualization": r"(?:可视化|图表)[：:]\s*(\d+(?:\.\d+)?)",
        "writing_quality": r"(?:写作质量|文字表达)[：:]\s*(\d+(?:\.\d+)?)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            scores[key] = float(match.group(1))

    return scores


@dataclass
class PaperScore:
    """论文评分"""
    total_score: float
    breakdown: Dict[str, float]
    strengths: List[str]
    improvements: List[str]
    raw_review: str


def parse_paper_review(text: str) -> PaperScore:
    """解析论文评审结果"""
    total_score = parse_score(text) or 0.0
    breakdown = parse_scoring_breakdown(text)

    # 提取优点
    strengths = []
    strengths_pattern = r"优点[：:](.*?)(?=改进|缺点|$)"
    match = re.search(strengths_pattern, text, re.DOTALL)
    if match:
        strength_text = match.group(1)
        # 按数字序号或换行分割
        items = re.split(r"\d+[\.、]|\n-\s*|\n\*\s*", strength_text)
        strengths = [item.strip() for item in items if item.strip()]

    # 提取改进方向
    improvements = []
    improve_pattern = r"(?:改进方向|改进建议|缺点)[：:](.*?)(?=优点|$)"
    match = re.search(improve_pattern, text, re.DOTALL)
    if match:
        improve_text = match.group(1)
        items = re.split(r"\d+[\.、]|\n-\s*|\n\*\s*", improve_text)
        improvements = [item.strip() for item in items if item.strip()]

    return PaperScore(
        total_score=total_score,
        breakdown=breakdown,
        strengths=strengths,
        improvements=improvements,
        raw_review=text
    )


def format_prompt_with_context(
    template: str,
    context: Dict[str, str]
) -> str:
    """用上下文格式化提示词模板"""
    result = template
    for key, value in context.items():
        placeholder = f"{{{key}}}"
        result = result.replace(placeholder, value)
    return result


def setup_work_directory(
    base_dir: str,
    work_dir_name: str,
    source_dirs: List[str],
    source_files: List[str]
) -> str:
    """设置工作目录"""
    work_dir = os.path.join(base_dir, work_dir_name)

    # 创建工作目录
    create_directory(work_dir)

    # 复制源目录
    for src_dir in source_dirs:
        src_path = os.path.join(base_dir, src_dir)
        if directory_exists(src_path):
            dst_path = os.path.join(work_dir, src_dir)
            if not directory_exists(dst_path):
                copy_directory(src_path, dst_path)

    # 复制源文件
    for src_file in source_files:
        src_path = os.path.join(base_dir, src_file)
        if file_exists(src_path):
            dst_path = os.path.join(work_dir, src_file)
            if not file_exists(dst_path):
                copy_file(src_path, dst_path)

    return work_dir


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate_weighted_score(
    scores: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """计算加权分数"""
    total = 0.0
    weight_sum = 0.0

    for key, score in scores.items():
        if key in weights:
            total += score * weights[key]
            weight_sum += weights[key]

    if weight_sum > 0:
        return total / weight_sum
    return 0.0


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total_steps: int, log_file: str = None):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_names: List[str] = []
        self.step_times: List[Tuple[str, str]] = []
        self.log_file = log_file

    def start_step(self, name: str):
        """开始一个步骤"""
        self.current_step += 1
        self.step_names.append(name)
        start_time = get_timestamp()
        self.step_times.append((start_time, ""))

        progress = self.current_step / self.total_steps * 100
        msg = f"[{self.current_step}/{self.total_steps}] ({progress:.1f}%) 开始: {name}"
        logging.info(msg)

        if self.log_file:
            append_file(self.log_file, f"{start_time} - {msg}\n")

    def end_step(self, success: bool = True, message: str = ""):
        """结束当前步骤"""
        end_time = get_timestamp()
        if self.step_times:
            start_time, _ = self.step_times[-1]
            self.step_times[-1] = (start_time, end_time)

        status = "完成" if success else "失败"
        name = self.step_names[-1] if self.step_names else "未知步骤"
        msg = f"[{self.current_step}/{self.total_steps}] {status}: {name}"
        if message:
            msg += f" - {message}"

        if success:
            logging.info(msg)
        else:
            logging.error(msg)

        if self.log_file:
            append_file(self.log_file, f"{end_time} - {msg}\n")

    def get_progress(self) -> Dict[str, Any]:
        """获取进度信息"""
        return {
            "current": self.current_step,
            "total": self.total_steps,
            "percentage": self.current_step / self.total_steps * 100,
            "steps": list(zip(self.step_names, self.step_times))
        }


# ============== LaTeX 编译相关函数 ==============

import subprocess
import platform
from dataclasses import dataclass as latex_dataclass


@dataclass
class LaTeXCompileResult:
    """LaTeX编译结果"""
    success: bool
    pdf_path: Optional[str]
    log_output: str
    error_message: str = ""


def find_latex_compiler() -> Optional[str]:
    """查找可用的LaTeX编译器

    Returns:
        编译器命令名称，如果未找到返回None
    """
    # 按优先级尝试不同的编译器
    compilers = ["xelatex", "pdflatex", "lualatex"]

    for compiler in compilers:
        try:
            # Windows 使用 where，Unix 使用 which
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["where", compiler],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:
                result = subprocess.run(
                    ["which", compiler],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

            if result.returncode == 0:
                logging.info(f"找到LaTeX编译器: {compiler}")
                return compiler
        except Exception:
            continue

    logging.warning("未找到可用的LaTeX编译器")
    return None


def compile_latex(
    tex_file: str,
    output_dir: Optional[str] = None,
    compiler: Optional[str] = None,
    runs: int = 2,
    timeout: int = 120
) -> LaTeXCompileResult:
    """编译LaTeX文件生成PDF

    Args:
        tex_file: .tex文件的完整路径
        output_dir: 输出目录，默认与tex文件同目录
        compiler: 编译器命令，默认自动检测
        runs: 编译次数（处理交叉引用通常需要2次）
        timeout: 单次编译超时时间（秒）

    Returns:
        LaTeXCompileResult 编译结果对象
    """
    if not file_exists(tex_file):
        return LaTeXCompileResult(
            success=False,
            pdf_path=None,
            log_output="",
            error_message=f"文件不存在: {tex_file}"
        )

    # 确定编译器
    if compiler is None:
        compiler = find_latex_compiler()
        if compiler is None:
            return LaTeXCompileResult(
                success=False,
                pdf_path=None,
                log_output="",
                error_message="未找到可用的LaTeX编译器，请安装TeX Live或MiKTeX"
            )

    # 确定工作目录和输出目录
    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_filename = os.path.basename(tex_file)
    tex_basename = os.path.splitext(tex_filename)[0]

    if output_dir is None:
        output_dir = tex_dir

    # 确保输出目录存在
    create_directory(output_dir)

    # 构建编译命令
    cmd = [
        compiler,
        "-interaction=nonstopmode",  # 非交互模式，遇错不停
        "-file-line-error",  # 显示文件行号
        f"-output-directory={output_dir}",
        tex_filename
    ]

    # 如果使用xelatex，添加支持中文的选项
    if compiler == "xelatex":
        # xelatex 默认支持 UTF-8 和中文
        pass

    all_output = []
    last_error = ""

    try:
        for run_num in range(1, runs + 1):
            logging.info(f"LaTeX编译第 {run_num}/{runs} 次: {tex_filename}")

            result = subprocess.run(
                cmd,
                cwd=tex_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )

            all_output.append(f"=== 第 {run_num} 次编译 ===\n")
            all_output.append(result.stdout)
            if result.stderr:
                all_output.append(f"\nSTDERR:\n{result.stderr}")

            # 检查是否成功
            if result.returncode != 0:
                # 尝试从日志中提取错误信息
                log_file = os.path.join(output_dir, f"{tex_basename}.log")
                if file_exists(log_file):
                    log_content = read_file(log_file) or ""
                    # 提取错误行
                    error_lines = []
                    for line in log_content.split('\n'):
                        if '!' in line or 'Error' in line or 'error' in line:
                            error_lines.append(line)
                    if error_lines:
                        last_error = '\n'.join(error_lines[:10])  # 最多10行错误
                    else:
                        last_error = result.stdout[-1000:] if result.stdout else "未知错误"
                else:
                    last_error = result.stdout[-1000:] if result.stdout else "编译失败"

                # 第一次编译失败就退出
                if run_num == 1:
                    return LaTeXCompileResult(
                        success=False,
                        pdf_path=None,
                        log_output='\n'.join(all_output),
                        error_message=last_error
                    )

        # 检查PDF是否生成
        pdf_path = os.path.join(output_dir, f"{tex_basename}.pdf")
        if file_exists(pdf_path):
            logging.info(f"PDF生成成功: {pdf_path}")
            return LaTeXCompileResult(
                success=True,
                pdf_path=pdf_path,
                log_output='\n'.join(all_output)
            )
        else:
            return LaTeXCompileResult(
                success=False,
                pdf_path=None,
                log_output='\n'.join(all_output),
                error_message="编译完成但未生成PDF文件"
            )

    except subprocess.TimeoutExpired:
        return LaTeXCompileResult(
            success=False,
            pdf_path=None,
            log_output='\n'.join(all_output),
            error_message=f"编译超时（{timeout}秒）"
        )
    except Exception as e:
        return LaTeXCompileResult(
            success=False,
            pdf_path=None,
            log_output='\n'.join(all_output),
            error_message=f"编译异常: {str(e)}"
        )


def compile_latex_with_bibtex(
    tex_file: str,
    output_dir: Optional[str] = None,
    compiler: Optional[str] = None,
    timeout: int = 120
) -> LaTeXCompileResult:
    """编译LaTeX文件（包含BibTeX处理）

    完整流程: latex -> bibtex -> latex -> latex

    Args:
        tex_file: .tex文件的完整路径
        output_dir: 输出目录
        compiler: 编译器命令
        timeout: 单次编译超时时间

    Returns:
        LaTeXCompileResult 编译结果对象
    """
    if not file_exists(tex_file):
        return LaTeXCompileResult(
            success=False,
            pdf_path=None,
            log_output="",
            error_message=f"文件不存在: {tex_file}"
        )

    # 确定编译器
    if compiler is None:
        compiler = find_latex_compiler()
        if compiler is None:
            return LaTeXCompileResult(
                success=False,
                pdf_path=None,
                log_output="",
                error_message="未找到可用的LaTeX编译器"
            )

    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_filename = os.path.basename(tex_file)
    tex_basename = os.path.splitext(tex_filename)[0]

    if output_dir is None:
        output_dir = tex_dir

    create_directory(output_dir)

    all_output = []

    try:
        # 第一次 LaTeX 编译
        logging.info(f"LaTeX编译 (1/4): {tex_filename}")
        cmd_latex = [
            compiler,
            "-interaction=nonstopmode",
            "-file-line-error",
            f"-output-directory={output_dir}",
            tex_filename
        ]

        result = subprocess.run(
            cmd_latex, cwd=tex_dir, capture_output=True,
            text=True, timeout=timeout, encoding='utf-8', errors='replace'
        )
        all_output.append(f"=== LaTeX 第1次 ===\n{result.stdout}")

        # BibTeX 编译
        aux_file = os.path.join(output_dir, f"{tex_basename}.aux")
        if file_exists(aux_file):
            logging.info(f"BibTeX编译 (2/4): {tex_basename}")
            cmd_bibtex = ["bibtex", os.path.join(output_dir, tex_basename)]

            result_bib = subprocess.run(
                cmd_bibtex, cwd=tex_dir, capture_output=True,
                text=True, timeout=timeout, encoding='utf-8', errors='replace'
            )
            all_output.append(f"=== BibTeX ===\n{result_bib.stdout}")

        # 第二次 LaTeX 编译
        logging.info(f"LaTeX编译 (3/4): {tex_filename}")
        result = subprocess.run(
            cmd_latex, cwd=tex_dir, capture_output=True,
            text=True, timeout=timeout, encoding='utf-8', errors='replace'
        )
        all_output.append(f"=== LaTeX 第2次 ===\n{result.stdout}")

        # 第三次 LaTeX 编译
        logging.info(f"LaTeX编译 (4/4): {tex_filename}")
        result = subprocess.run(
            cmd_latex, cwd=tex_dir, capture_output=True,
            text=True, timeout=timeout, encoding='utf-8', errors='replace'
        )
        all_output.append(f"=== LaTeX 第3次 ===\n{result.stdout}")

        # 检查PDF
        pdf_path = os.path.join(output_dir, f"{tex_basename}.pdf")
        if file_exists(pdf_path):
            logging.info(f"PDF生成成功: {pdf_path}")
            return LaTeXCompileResult(
                success=True,
                pdf_path=pdf_path,
                log_output='\n'.join(all_output)
            )
        else:
            return LaTeXCompileResult(
                success=False,
                pdf_path=None,
                log_output='\n'.join(all_output),
                error_message="编译完成但未生成PDF文件"
            )

    except subprocess.TimeoutExpired:
        return LaTeXCompileResult(
            success=False,
            pdf_path=None,
            log_output='\n'.join(all_output),
            error_message=f"编译超时（{timeout}秒）"
        )
    except Exception as e:
        return LaTeXCompileResult(
            success=False,
            pdf_path=None,
            log_output='\n'.join(all_output),
            error_message=f"编译异常: {str(e)}"
        )


def clean_latex_auxiliary_files(
    tex_file: str,
    output_dir: Optional[str] = None
) -> List[str]:
    """清理LaTeX编译产生的辅助文件

    Args:
        tex_file: .tex文件路径
        output_dir: 输出目录

    Returns:
        被删除的文件列表
    """
    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_basename = os.path.splitext(os.path.basename(tex_file))[0]

    if output_dir is None:
        output_dir = tex_dir

    # 要清理的扩展名
    aux_extensions = [
        '.aux', '.log', '.out', '.toc', '.lof', '.lot',
        '.bbl', '.blg', '.nav', '.snm', '.vrb',
        '.fdb_latexmk', '.fls', '.synctex.gz'
    ]

    deleted_files = []

    for ext in aux_extensions:
        aux_path = os.path.join(output_dir, f"{tex_basename}{ext}")
        if file_exists(aux_path):
            try:
                os.remove(aux_path)
                deleted_files.append(aux_path)
            except Exception as e:
                logging.warning(f"无法删除文件 {aux_path}: {e}")

    return deleted_files
