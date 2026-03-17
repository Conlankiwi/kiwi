// Qwen API 调用示例 - Node.js版本

const axios = require('axios');

// 配置
const API_KEY = "sk-6c639c66659d4c0ba8b8679da9723a5e";  // 替换为你的真实API Key
const BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1";
const MODEL = "qwen3.5-plus";

/**
 * 调用Qwen API
 * @param {string} message - 用户消息
 * @param {Array} history - 对话历史
 * @returns {Promise<string>} - AI回复
 */
async function callQwen(message, history = []) {
    // 构建消息列表
    const messages = [...history, { role: "user", content: message }];
    
    try {
        const response = await axios.post(
            `${BASE_URL}/chat/completions`,
            {
                model: MODEL,
                messages: messages,
                max_tokens: 2048,
                temperature: 0.7,
                stream: false
            },
            {
                headers: {
                    'Authorization': `Bearer ${API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 30000
            }
        );
        
        return response.data.choices[0].message.content;
    } catch (error) {
        throw new Error(`API调用失败: ${error.message}`);
    }
}

/**
 * 代码生成示例
 */
async function generateCode() {
    console.log('='.repeat(50));
    console.log('Qwen 代码生成助手');
    console.log('='.repeat(50));
    
    const prompt = `请帮我创建一个 Python 函数，功能是：读取一个 JSON 文件，统计其中每个键出现的次数，返回统计结果字典。要求：
1. 使用标准库，不需要额外依赖
2. 添加适当的错误处理
3. 添加函数文档字符串`;
    
    console.log('\n请求：生成代码...\n');
    
    try {
        const code = await callQwen(prompt);
        console.log('Qwen 生成的代码：\n');
        console.log(code);
    } catch (error) {
        console.error('错误：', error.message);
    }
}

// 运行示例
generateCode();
