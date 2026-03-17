#!/usr/bin/env python3
"""
Qwen API 对话机器人
=================
一个支持多轮对话的交互式命令行聊天机器人，通过阿里云百炼平台调用 Qwen 大模型 API。

功能特性:
    - 多轮对话：自动维护对话历史上下文
    - 错误处理：完整的网络异常和 API 错误处理
    - 交互界面：支持退出、清空历史、显示历史等命令
    - 配置管理：API 密钥和参数集中配置

依赖:
    pip install requests

使用方法:
    python3 qwen_chat.py

作者：Qwen Code
日期：2026-03-17
"""

import requests
import json
import sys
from typing import Optional, List, Dict, Any


# =============================================================================
# 配置区域 - 在此处修改 API 密钥和模型参数
# =============================================================================

# 百炼平台 API Key (建议通过环境变量 DASHSCOPE_API_KEY 设置)
API_KEY = "sk-6c639c66659d4c0ba8b8679da9723a5e"

# API 基础 URL - 阿里云百炼平台兼容模式端点
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型名称 - 可选值：qwen-turbo, qwen-plus, qwen-max, qwen3.5-plus 等
MODEL = "qwen3.5-plus"

# 请求超时时间 (秒)
TIMEOUT = 30

# 最大对话历史轮数 (超过后自动清除最早的历史)
MAX_HISTORY_TURNS = 20


# =============================================================================
# API 调用函数
# =============================================================================

def call_qwen_api(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Dict[str, Any]:
    """
    调用 Qwen API 发送对话请求
    
    该函数负责构建 HTTP 请求并发送到百炼平台，处理响应并返回结果。
    包含完整的错误处理机制，包括网络错误、认证错误、速率限制等。

    Args:
        messages: 消息列表，每个消息是包含 'role' 和 'content' 的字典
                  role 可以是 'system', 'user', 'assistant'
        temperature: 生成温度，控制随机性 (0.0-2.0)，越高越随机
        max_tokens: 最大生成 token 数

    Returns:
        包含 API 响应的字典，结构为：
        {
            "success": bool,      # 请求是否成功
            "content": str,       # AI 回复内容 (成功时)
            "error": str,         # 错误信息 (失败时)
            "usage": dict         # token 使用统计 (成功时)
        }

    Raises:
        不抛出异常，所有错误都封装在返回字典中
    """
    # 构建请求头
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Qwen-ChatBot/1.0"
    }

    # 构建请求体
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False  # 非流式响应
    }

    # 发送 HTTP POST 请求
    try:
        response = requests.post(
            url=f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )

        # 检查 HTTP 状态码
        response.raise_for_status()

        # 解析 JSON 响应
        result = response.json()

        # 提取 AI 回复内容
        choices = result.get("choices", [])
        if not choices:
            return {
                "success": False,
                "error": "API 响应中没有 choices 字段",
                "raw_response": result
            }

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # 提取 token 使用统计
        usage = result.get("usage", {})

        return {
            "success": True,
            "content": content,
            "usage": usage
        }

    # 处理各种 HTTP 请求异常
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"请求超时 (>{TIMEOUT}秒)，请检查网络连接"
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "网络连接失败，请检查网络或 API 端点"
        }
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "未知"
        error_msg = parse_api_error(e.response) if e.response else str(e)
        return {
            "success": False,
            "error": f"HTTP 错误 {status_code}: {error_msg}"
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"请求异常：{str(e)}"
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"响应解析失败：{str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"未知错误：{str(e)}"
        }


def parse_api_error(response) -> str:
    """
    解析 API 返回的错误信息
    
    尝试从响应体中提取结构化的错误消息，如果解析失败则返回状态码。

    Args:
        response: requests 响应对象

    Returns:
        错误信息字符串
    """
    try:
        error_data = response.json()
        error = error_data.get("error", {})
        if isinstance(error, dict):
            return error.get("message", str(error))
        return str(error)
    except (json.JSONDecodeError, AttributeError):
        return f"状态码：{response.status_code}"


# =============================================================================
# 对话管理类
# =============================================================================

class ChatSession:
    """
    对话会话管理类
    
    负责维护对话历史、管理上下文、限制历史长度。
    支持系统提示词设置和多轮对话记忆。
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        初始化对话会话

        Args:
            system_prompt: 系统提示词，用于设定 AI 角色和行为准则
                          如果为 None，则使用默认提示词
        """
        self.messages: List[Dict[str, str]] = []
        
        # 设置系统提示词
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            # 默认系统提示词：设定 AI 为友好助手
            self.messages.append({
                "role": "system",
                "content": "你是一个友好、乐于助人的 AI 助手。请用简洁、准确的语言回答用户问题。"
            })

    def add_user_message(self, content: str) -> None:
        """
        添加用户消息到对话历史

        Args:
            content: 用户消息内容
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """
        添加 AI 回复到对话历史

        Args:
            content: AI 回复内容
        """
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """
        获取完整的消息历史（用于 API 调用）

        Returns:
            消息列表
        """
        return self.messages

    def get_turn_count(self) -> int:
        """
        获取对话轮数（一轮 = 用户消息 + AI 回复）

        Returns:
            对话轮数
        """
        # 减去 system 消息，然后除以 2
        return (len(self.messages) - 1) // 2

    def trim_history(self, max_turns: int) -> int:
        """
        修剪对话历史，保留最近的 N 轮对话
        
        当对话过长时调用此方法，保留 system 消息和最近的 max_turns 轮对话。

        Args:
            max_turns: 最大保留轮数

        Returns:
            被移除的消息数量
        """
        if len(self.messages) <= 1:
            return 0

        # 计算需要保留的消息数 (system + 2 * max_turns)
        keep_count = 1 + (max_turns * 2)
        
        if len(self.messages) <= keep_count:
            return 0

        # 保留 system 消息和最近的对话
        removed_count = len(self.messages) - keep_count
        self.messages = [self.messages[0]] + self.messages[-(keep_count - 1):]
        
        return removed_count

    def clear_history(self) -> None:
        """
        清空对话历史，保留系统提示词
        """
        system_message = self.messages[0] if self.messages else None
        self.messages = []
        if system_message:
            self.messages.append(system_message)

    def export_history(self) -> str:
        """
        导出对话历史为 JSON 字符串

        Returns:
            JSON 格式的对话历史
        """
        return json.dumps(self.messages, ensure_ascii=False, indent=2)


# =============================================================================
# 命令行界面
# =============================================================================

def print_banner() -> None:
    """打印程序启动横幅"""
    print("=" * 60)
    print(" " * 18 + "Qwen API 对话机器人")
    print("=" * 60)
    print(f"  模型：{MODEL}")
    print(f"  最大历史轮数：{MAX_HISTORY_TURNS}")
    print("-" * 60)


def print_help() -> None:
    """打印帮助信息"""
    print("""
可用命令:
  quit, exit, q     - 退出程序
  clear, reset      - 清空对话历史
  history, log      - 显示当前对话历史
  help, h           - 显示此帮助信息
  count             - 显示对话轮数统计

其他:
  直接输入文字即可与 AI 对话
  按 Ctrl+C 可强制退出
""")


def print_statistics(session: ChatSession) -> None:
    """
    打印对话统计信息

    Args:
        session: 当前对话会话
    """
    turn_count = session.get_turn_count()
    message_count = len(session.get_messages()) - 1  # 减去 system 消息
    print(f"\n[统计] 当前对话：{turn_count} 轮，共 {message_count} 条消息")


def interactive_chat() -> None:
    """
    主循环：交互式命令行对话
    
    持续读取用户输入，调用 API 获取回复，直到用户选择退出。
    支持特殊命令和错误恢复。
    """
    print_banner()
    print_help()

    # 创建对话会话
    session = ChatSession()

    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 你：").strip()

            # 处理空输入
            if not user_input:
                continue

            # 处理命令（不区分大小写）
            cmd = user_input.lower()

            if cmd in ('quit', 'exit', 'q'):
                print("\n👋 再见！感谢使用 Qwen 对话机器人")
                break

            elif cmd in ('clear', 'reset'):
                session.clear_history()
                print("✅ 对话历史已清空")
                continue

            elif cmd in ('history', 'log'):
                print("\n" + "-" * 40)
                print(session.export_history())
                print("-" * 40)
                continue

            elif cmd in ('help', 'h'):
                print_help()
                continue

            elif cmd == 'count':
                print_statistics(session)
                continue

            # 添加用户消息到会话
            session.add_user_message(user_input)

            # 调用 API
            print("\n🤖 Qwen: ", end="", flush=True)

            result = call_qwen_api(session.get_messages())

            if result["success"]:
                # 显示 AI 回复
                response_content = result["content"]
                print(response_content)

                # 添加 AI 回复到会话历史
                session.add_assistant_message(response_content)

                # 检查是否需要修剪历史
                if session.get_turn_count() > MAX_HISTORY_TURNS:
                    removed = session.trim_history(MAX_HISTORY_TURNS)
                    if removed > 0:
                        print(f"\n[提示] 已自动移除 {removed} 条早期消息以保持上下文长度")

                # 显示 token 使用统计（可选）
                usage = result.get("usage", {})
                if usage:
                    total_tokens = usage.get("total_tokens", 0)
                    print(f"\n[Token 使用] 本次消耗：{total_tokens}")

            else:
                # API 调用失败
                print(f"\n❌ 错误：{result['error']}")
                # 移除刚才添加的用户消息（因为请求失败了）
                session.messages.pop()

        except KeyboardInterrupt:
            # 处理 Ctrl+C 中断
            print("\n\n⚠️  检测到中断信号")
            confirm = input("确定要退出吗？(y/n): ").strip().lower()
            if confirm in ('y', 'yes', '是'):
                print("\n👋 再见！")
                break
            else:
                continue

        except EOFError:
            # 处理输入流结束（如管道输入）
            print("\n\n👋 输入结束，退出程序")
            break

        except Exception as e:
            # 捕获其他未预期的异常
            print(f"\n❌ 程序异常：{str(e)}")
            print("建议：请检查输入或重启程序")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    # 检查依赖
    try:
        import requests
    except ImportError:
        print("❌ 错误：缺少 requests 库")
        print("请运行：pip install requests")
        sys.exit(1)

    # 启动交互式对话
    interactive_chat()
