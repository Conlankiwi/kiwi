#!/usr/bin/env python3
"""
Qwen API 调用示例 - 对话机器人
"""

import requests
import json

# 配置
API_KEY = "sk-6c639c66659d4c0ba8b8679da9723a5e"  # 替换为你的真实API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.5-plus"

def chat_with_qwen(message, history=None):
    """
    调用 Qwen API 进行对话
    
    Args:
        message: 用户消息
        history: 对话历史列表（可选）
    
    Returns:
        AI 回复内容
    """
    # 构建请求头
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构建消息列表
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    
    # 构建请求体
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        # 发送请求
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()  # 检查HTTP错误
        
        # 解析响应
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        return f"API 调用失败: {str(e)}"
    except KeyError as e:
        return f"响应解析失败: {str(e)}"

def main():
    """主函数：交互式对话"""
    print("=" * 50)
    print("Qwen API 对话机器人")
    print("=" * 50)
    print("输入 'quit' 退出，输入 'clear' 清空历史")
    print()
    
    history = []
    
    while True:
        # 获取用户输入
        user_input = input("\n你: ").strip()
        
        if user_input.lower() == 'quit':
            print("再见！")
            break
        elif user_input.lower() == 'clear':
            history = []
            print("对话历史已清空")
            continue
        elif not user_input:
            continue
        
        # 调用API
        print("\nQwen: ", end="", flush=True)
        response = chat_with_qwen(user_input, history)
        print(response)
        
        # 保存到历史记录
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
