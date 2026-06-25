# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
数学答案判分与奖励函数模块
来源：https://github.com/sail-sg/understand-r1-zero/blob/main/understand_r1_zero/math_grader.py

在CS336推理对齐训练流水线中的定位：
    这是SFT评估、Expert Iteration筛选、RLHF奖励计算的核心「裁判」组件
    负责判断模型生成的数学答案是否正确，输出可用于训练的奖励信号

设计核心思想：
    大模型生成的数学答案格式千差万别（LaTeX写法不统一、带单位、带多余文本、格式变体多），
    单一的字符串精确匹配会大量误判正确答案为错误。因此采用「多层级归一化 + 多维度判分」的架构：
    1. 字符串级归一化：消除格式、单位、LaTeX写法差异，做快速精确匹配
    2. 数值级判分：对数值答案做近似相等判断，容忍浮点精度误差
    3. 符号级判分：用SymPy做代数化简，判断数学表达式语义等价
    4. 专用工具判分：调用math_verify库做高召回的数学验证
    同时加入超时保护、重复内容检测，保证评估的鲁棒性与效率
"""

import re
import signal
from itertools import islice, zip_longest
from math import isclose
from typing import Optional

import sympy
from latex2sympy2_extended import latex2sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from pylatexenc import latex2text
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


# =============================================================================
# 第一部分：基础答案归一化工具（MATH数据集经典实现）
# =============================================================================

def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    """
    MATH数据集标准答案归一化函数（Dan Hendrycks 经典实现）
    作用：剥离外层文本包装、统一LaTeX格式、去除单位与无关符号，得到纯净的答案表达式
    是速度最快的一级判分依据，能覆盖大部分格式差异场景

    Args:
        answer: 原始答案字符串，可能带LaTeX、单位、文本包装

    Returns:
        归一化后的答案字符串；输入为None则返回None
    """
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # 剥离最外层的 \text{...} 包装，提取纯文本内容
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        # 执行深度字符串清洗与格式统一
        return _strip_string(answer)
    except:
        # 任何异常都直接返回原始答案，保证鲁棒性
        return answer


# 单位词列表（主要来自MathQA数据集）
# 用于从答案中剥离单位，只比对数学数值部分
unit_texts = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m",
    "deg", "mile", "q .", "monkey", "prime", "ratio", "profit of rs",
    "rd", "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain",
    "ab", "way", "west", "a .", "b .", "c .", "d .", "e .", "f .", "g .",
    "h .", "t", "a", "h", "no change", "men", "soldier", "pie", "bc",
    "excess", "st", "inches", "noon", "percent", "by", "gal", "kmh", "c",
    "acre", "rise", "a . m", "th", "π r 2", "sq", "mark", "l", "toy",
    "coin", "sq . m", "gallon", "° f", "profit", "minw", "yr", "women",
    "feet", "am", "pm", "hr", "cu cm", "square", "v â € ™", "are", "rupee",
    "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day",
    "hour", "minute", "min", "second", "man", "woman", "sec", "cube", "mt",
    "sq inch", "mp", "∏ cm ³", "hectare", "more", "sec", "unit", "cu . m",
    "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m", "cm", "mm",
    "apple", "liter", "loss", "yard", "pure", "year", "increase", "decrease",
    "d", "less", "Surface", "litre", "pi sq m", "s .", "metre", "meter", "inch",
]
# 补充单位的复数形式，覆盖更多写法
unit_texts.extend([t + "s" for t in unit_texts])


def _strip_string(string):
    """
    内部工具：对答案字符串做深度清洗与LaTeX格式统一
    核心目标：消除所有不影响数学语义的格式差异，让等价答案的字符串完全一致
    """

    def _fix_fracs(string):
        """统一分数写法：将简写的 \frac12 补全为标准 \frac{1}{2} 格式"""
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    # 已经是标准格式，直接保留
                    new_str += substr
                else:
                    # 处理 \fracab 这种无括号的简写
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        # 两个字符都不是括号，补全为 \frac{a}{b}
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        # 第一个字符无括号，第二个有括号，补全第一个
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        """将 a/b 形式的斜杠分数，转换为标准LaTeX分数 \frac{a}{b}"""
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == f"{a}/{b}"
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        """剥离右侧用 \text{ } 包裹的单位描述"""
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        """统一根号写法：将 \sqrt3 简写补全为标准 \sqrt{3} 格式"""
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # ---------- 逐步骤清洗 ----------
    # 去除换行符
    string = string.replace("\n", "")
    # 去除LaTeX负间距命令 \!
    string = string.replace("\\!", "")
    # 统一转义反斜杠：将双反斜杠替换为单反斜杠
    string = string.replace("\\\\", "\\")

    # 统一矩阵环境：将 array、bmatrix 统一为 pmatrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # 统一分数命令：tfrac/dfrac 都替换为标准 frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # 统一关系符号缩写：\neq→\ne、\leq→\le、\geq→\ge
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # 去除 \left 和 \right 可伸缩括号命令，不影响数学语义
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # 去除末尾的 \text{...} 单位文本
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    # 两轮遍历：剥离所有单位词
    for _ in range(2):
        for unit_text in unit_texts:
            # 匹配独立的单位词（前后为非字母字符或字符串首尾），替换为空
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # 去除角度符号 ^\circ
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # 去除美元符号
    string = string.replace("\\$", "")

    # 再次去除右侧文本单位
    string = _remove_right_units(string)

    # 去除百分号
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # 小数点补零：将 ".5" 补为 "0.5"，"{.5}" 补为 "{0.5}"
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # 去除开头的短变量赋值，如 "x = 5" → "5"
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # 修复根号简写
    string = _fix_sqrt(string)

    # 去除所有空格
    string = string.replace(" ", "")

    # 修复分数简写
    string = _fix_fracs(string)

    # 特殊值统一：0.5 → \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # 斜杠分数转LaTeX分数
    string = _fix_a_slash_b(string)

    return string


# =============================================================================
# 第二部分：深度答案归一化（来自论文《Training Verifiers to Solve Math Word Problems》）
# =============================================================================

# 替换规则：去除冗余冠词、符号，统一文本命令
SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""),
    (r"\ ", ""), (" ", ""), ("mbox", "text"),
    (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

# 移除词列表：去除答案中出现的无关名词、单位、语气词
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
    "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    深度答案归一化（来自论文 https://arxiv.org/pdf/2206.14858.pdf 第18页）
    比 mathd_normalize_answer 更激进，剥离更多包装格式，用于二级字符串匹配

    处理范围：
    - 去除冗余文本、冠词、单位
    - 提取 $...$、\boxed、\textbf、\overline 包裹的核心答案
    - 统一LaTeX简写语法
    - 去除千分位逗号
    """
    # 第一轮：替换所有预定义的子串
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    # 第二轮：移除所有无关表达
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # 提取LaTeX数学环境、加粗、方框、overline包裹的核心答案内容
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)

    # 统一LaTeX简写：\fracab → \frac{a}{b}、\sqrta → \sqrt{a}
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # 去除千分位逗号：100,000 → 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


# =============================================================================
# 第三部分：鲁棒性辅助工具
# =============================================================================

def repeatness(s: str):
    """
    检测字符串是否存在高度重复的模式（循环输出）
    原理：基于后缀数组计算最长公共前缀总和，判断重复占比
    作用：推理模型容易出现循环重复输出，这类内容直接判错，避免后续sympy解析卡死

    Returns:
        bool: 重复度超过阈值返回True，视为无效输出
    """
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        """构建后缀数组，用于高效计算重复度"""
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        """计算最长公共前缀数组"""
        n, ans, k = len(arr), [0] * len(arr), 0
        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue
            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1
            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1
        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    # 重复度超过20%则判定为循环输出
    return (cnt * 2 / (n * (n + 1))) > 0.2


class timeout:
    """
    超时控制上下文管理器
    工程意义：SymPy解析复杂LaTeX可能陷入长时间计算甚至卡死，
    用信号量强制超时，保证评估流程不会被单个异常样本阻塞

    用法：
        with timeout(seconds=1):
            # 可能超时的代码
    """
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# =============================================================================
# 第四部分：数学等价判断工具
# =============================================================================

def latex_eval(latex):
    """解析LaTeX表达式为SymPy对象，并计算其数值"""
    sym = parse_latex(latex)
    val = sym.evalf()
    return sym, val


def numeric_equal(prediction: float, reference: float):
    """
    数值近似相等判断
    使用相对误差1e-4，容忍浮点计算的精度差异，避免因计算精度导致正确答案被误判
    """
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    """
    符号级等价判断：尝试多种方式验证两个数学表达式语义等价
    从快到慢依次尝试：直接相等 → 化简后相等 → 等式左右差相等 → 数值近似相等 → 矩阵逐元素相等
    所有方式都失败才返回False，最大化召回率
    """
    def _parse(s):
        """尝试多种解析器解析表达式，只要有一种成功即可"""
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # 1. 直接字符串/对象相等
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # 2. 代数化简后差值为0
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # 3. 等式形式：左右两边差值的绝对值相等
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    # 4. 数值近似相等
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # 5. 矩阵类型：逐元素四舍五入后比较
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def _is_latex_equal(str1, str2):
    """内部LaTeX相等判断：先直接解析，失败则归一化后再解析，最后退化为字符串比较"""
    try:
        sym1, val1 = latex_eval(str1)
        sym2, val2 = latex_eval(str2)
        if sym1 == sym2 or val1 == val2:
            return True
        else:
            raise ValueError
    except Exception:
        try:
            norm1, norm2 = normalize_final_answer(str1), normalize_final_answer(str2)
            sym1, val1 = latex_eval(norm1)
            sym2, val2 = latex_eval(norm2)
            if sym1 == sym2 or val1 == val2:
                return True
        except Exception:
            return norm1 == norm2
    return False


def is_latex_equal(given_answer: str, ground_truth: str) -> bool:
    """
    高召回LaTeX答案判分（非fast模式使用）
    流程：重复检测 → 归一化字符串匹配 → math_verify库验证
    精度高但速度慢，用于最终评估或错误样本召回

    Args:
        given_answer: 模型生成的答案
        ground_truth: 标准答案

    Returns:
        bool: 是否判定为正确
    """
    try:
        with timeout(1):
            try:
                # 前置过滤：长文本且重复度高，直接判错
                if (len(given_answer) > 128 and repeatness(given_answer)) or (
                    len(ground_truth) > 128 and repeatness(ground_truth)
                ):
                    return False

                # 第一级：归一化后字符串精确匹配，最快
                ground_truth_normalized = _normalize(ground_truth)
                given_normalized = _normalize(given_answer)
                if ground_truth_normalized is None:
                    return False
                if ground_truth_normalized == given_normalized:
                    return True

                # 第二级：调用math_verify库做专业数学验证
                given_answer.replace("\n", "")
                ground_truth.replace("\n", "")
                # 确保答案被$包裹，符合math_verify的输入要求
                if "$" not in given_answer:
                    given_answer = f"${given_answer}$"
                if "$" not in ground_truth:
                    ground_truth = f"${ground_truth}$"

                return verify(
                    parse(
                        ground_truth,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    parse(
                        given_answer,
                        extraction_config=(
                            LatexExtractionConfig(boxed_match_priority=0),
                            ExprExtractionConfig(),
                        ),
                        fallback_mode="no_fallback",
                        extraction_mode=["first_match"],
                        parsing_timeout=1,
                    ),
                    timeout_seconds=1,
                )
            except Exception:
                return False
    except TimeoutError:
        return False


def is_value_equal(given_answer: str, ground_truth: str) -> bool:
    """基础数值相等判断：字符串相等 或 转为浮点数后相等"""
    assert ground_truth is not None
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    str_equal = ground_truth_normalized_mathd == given_answer_normalized_mathd
    try:
        number_equal = float(ground_truth_normalized_mathd) == float(
            given_answer_normalized_mathd
        )
        return str_equal or number_equal
    except Exception:
        return str_equal


# =============================================================================
# 第五部分：SymPy判分体系
# =============================================================================

# 危险子串/正则：SymPy解析这类表达式容易卡死，直接跳过
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """将表达式字符串解析为SymPy对象，支持隐式乘法、^替换为**"""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """将LaTeX表达式转换为SymPy可识别的纯文本表达式"""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # 适配带分数
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # 替换特殊数学符号为英文单词
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    """判断字符串是否为浮点数"""
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    """判断浮点数是否等价于整数"""
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    """判断字符串是否为分数形式"""
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    """判断字符串是否表示整数（支持千分位逗号）"""
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    """字符串转整数，自动去除千分位逗号"""
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    处理带分数：将 "7 3/4" 转换为 "7+3/4"，使其可被计算
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub(r"\1+\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    """去除千分位逗号，同时避免误删元组/列表中的逗号"""
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """
    SymPy判分专用的表达式归一化
    处理：文本包装、单位、大数字词、括号、带分数、空格、大小写、千分位等
    """
    if expr is None:
        return None

    # 去除外层\text{}
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    # 大数字词展开：million/billion/trillion
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    # 去除单位词及指数形式
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second",
        "minute", "hour", "day", "week", "month", "year", "foot",
        "feet", "inch", "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    # 去除最外层大括号
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)

    # 浮点整数值统一为整数形式
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))

    # LaTeX转纯文本表达式
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # 负号与空格处理
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # 残余的LaTeX括号直接去掉
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # 文本答案统一小写
    expr = expr.lower()

    # 整数字符串标准化
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    """统计表达式中未知字母的数量，用于判断是否适合用SymPy计算"""
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    """
    判断是否允许用SymPy求值
    过滤规则：未知字母超过2个、含危险幂次语法，都跳过求值，防止卡死
    """
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    """用SymPy化简判断两个表达式差值是否为0"""
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    拆分元组/区间答案为多个元素，同时正确处理千分位逗号
    例如 "(1, 2, 3)" → ["1", "2", "3"]
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


# =============================================================================
# 第六部分：答案提取与核心判分入口
# =============================================================================

def last_boxed_only_string(string):
    """
    找到字符串中**最后一个** \boxed 命令，并提取完整的 \boxed{...} 片段
    为什么取最后一个？因为模型可能中间出现多个boxed，最终答案通常在最后一个
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    # 括号匹配：找到对应闭合的右大括号
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    """剥离 \boxed{...} 的外层包装，只保留内部答案内容"""
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """从完整解答中提取 \boxed 包裹的最终答案"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    """
    SymPy体系判分：归一化后字符串匹配 + 元组拆分 + 符号化简验证
    比mathd体系更严格，能处理代数表达式、分数、多元答案等复杂场景
    """
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    # 字符串完全一致直接返回正确
    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    # 拆分元组/多答案，逐元素比对
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    # 元组长度/首尾不匹配直接判错
    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        is_correct = True
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # 分数要求完全一致，不做约分匹配，保证答案最简形式
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # 标准答案是整数而生成答案不是，严格判错
                is_correct = False
            else:
                # 其余情况用SymPy化简判断
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    """
    MATH标准快速判分：基于mathd_normalize_answer的字符串精确匹配
    速度最快，作为第一级判分，覆盖大部分常规场景
    """
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def extract_answer(passage: str) -> str:
    """
    统一答案提取入口
    优先提取 \boxed 包裹的答案；没有boxed则返回None
    """
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    """
    核心判分总入口
    两级判分策略：
        fast=True（训练评估用）：mathd快速匹配 + sympy符号匹配，速度快，召回率足够
        fast=False（最终测试用）：额外增加math_verify高召回验证，速度慢，准确率最高

    Args:
        model_answer: 模型生成的答案内容（已提取的纯答案，非完整解答）
        gt_answer: 标准答案
        fast: 是否使用快速模式

    Returns:
        bool: 是否正确
    """
    # 若标准答案带boxed，先提取内部内容
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)

    # 两级快速判分
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(
        model_answer, gt_answer
    )

    # 非快速模式：追加math_verify做高召回补全
    if not fast:
        correct = correct or is_latex_equal(model_answer, gt_answer)

    return correct


# =============================================================================
# 第七部分：奖励函数（用于SFT评估与Expert Iteration筛选）
# =============================================================================

def r1_zero_reward_fn(response, ground_truth, fast=True):
    """
    R1 风格严格奖励函数
    对应 DeepSeek R1 的 ...... 输出格式规范
    判分逻辑：
        1. 先检查格式：必须同时包含   和  标签
        2. 格式不达标：所有奖励均为0
        3. 格式达标：提取answer内的内容，再做答案正确性判分
        4. 答案正确：总奖励1.0；答案错误：总奖励0.0，但格式奖励仍为1.0

    适用场景：Expert Iteration 中筛选符合R1范式的正确推理轨迹
    """
    if "<answer>" in response and "</answer>" in response:
        # 提取标签内的内容
        model_answer = response.split("<answer>")[-1].replace("</answer>", "")

        # 提取boxed答案
        if "\\boxed" in model_answer:
            model_answer = extract_answer(model_answer)
            if model_answer is None:
                return {
                    "format_reward": 1.0,
                    "answer_reward": 0.0,
                    "reward": 0.0
                }

        # 标准答案类型兼容：支持数值、字符串、多正确答案列表
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        if isinstance(ground_truth, str):
            is_correct = grade(model_answer, ground_truth, fast)
        elif isinstance(ground_truth, list):
            is_correct = False
            for gt in ground_truth:
                is_correct |= grade(model_answer, gt, fast)

        if is_correct:
            return {
                "format_reward": 1.0,
                "answer_reward": 1.0,
                "reward": 1.0
            }
        else:
            # 格式正确但答案错误：格式给分，答案不给分，避免模型为了格式奖励作弊
            return {
                "format_reward": 1.0,
                "answer_reward": 0.0,
                "reward": 0.0
            }
    else:
        # 格式不达标，所有奖励为0
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }


def question_only_reward_fn(response, ground_truth, fast=True):
    """
    宽松版奖励函数
    不要求思考标签格式，只检查是否有 \boxed 答案并判分
    适用场景：基础SFT阶段评估、不限制输出格式的实验
    """
    model_answer = extract_answer(response)
    if model_answer is None:
        # 连boxed答案都提取不到，格式分0
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }

    # 兼容多种标准答案类型
    if isinstance(ground_truth, float) or isinstance(ground_truth, int):
        ground_truth = str(ground_truth)
    if isinstance(ground_truth, str):
        is_correct = grade(model_answer, ground_truth, fast)
    elif isinstance(ground_truth, list):
        is_correct = False
        for gt in ground_truth:
            is_correct |= grade(model_answer, gt, fast)

    if is_correct:
        return {
            "format_reward": 1.0,
            "answer_reward": 1.0,
            "reward": 1.0
        }
    else:
        # 有boxed但答案错误，格式给分
        return {
            "format_reward": 1.0,
            "answer_reward": 0.0,
            "reward": 0.0
        }