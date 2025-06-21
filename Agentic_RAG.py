from openai import OpenAI  
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import re
from datetime import date
import json
import os
import logging
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agentic_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

@dataclass 
class AgentConfig:
    """配置类，集中管理系统参数"""
    # 大语言模型相关超参
    model_name: str = 'deepseek-reasoner'
    temperature: float = 0.6
    top_p: float = 0.9
    # 不再建议使用frequency_penalty参数
    # frequency_penalty: float = 1.05 
    max_tokens: int = 4096
    
    # 工具使用相关参数
    max_search_times: int = 5  # 工具的最多使用次数
    each_search_max_result: int = 5  # 每次联网检索召回的文档数
    
    # API密钥
    openai_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    openai_base_url: str = "https://api.deepseek.com/beta"  # beta才有prefix功能
    
    # 重试配置
    retry_attempts: int = 3
    retry_min_wait: int = 2
    retry_max_wait: int = 10
    
    # 超时设置
    request_timeout: int = 600
    
    @property
    def system_prompt(self) -> str:
        """返回经过参数替换的系统提示词"""
        return f"""You are DeepSeek Assistant, an AI built for retrieval augmented generation.
You are at 2025 and current date is {date.today()}.
You have access to the web_search tool to retrieve relevant information to help answer user questions.
You can use web_search tool up to {self.max_search_times} times to answer a user's question, but try to be efficient and use as few as possible.

Below are some guidelines:
- Use web_search for general internet queries, like finding current events or factual information.
- Always provide a final answer in a clear and concise manner, with citations for any information obtained from the internet, indicate the source article title and article link in format like that: [article title](artile link).
- If you think you need to use a tool, format your response as a tool call with the `action` and `action_input` within <tool_call>...</tool_call>, like this:
<tool_call>
{{ "action": "web_search", "action_input": {{ "query": "current stock price of Tesla" }} }}
</tool_call>
- After using a tool, continue your reasoning based on the web_search result in <tool_response>...</tool_response>.
- Remember that if you need multi-turn web_search to find relevant information, make sure you conduct all search tasks before you provide a final answer.
- For complex questions, consider breaking them down into simpler sub-questions for more effective searches.
- Cite your sources clearly and provide balanced information from multiple reliable sources when available.
---
"""

# 创建单例配置对象
config = AgentConfig()

# 初始化客户端
def create_openai_client() -> OpenAI:
    """创建并返回OpenAI客户端，使用配置中的参数"""
    if not config.openai_api_key:
        raise ValueError("DEEPSEEK_API_KEY not found. Please set it in .env file or environment variables.")
    
    return OpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        timeout=config.request_timeout
    )

# 懒加载客户端
_client = None
def get_client() -> OpenAI:
    """懒加载方式获取OpenAI客户端"""
    global _client
    if _client is None:
        _client = create_openai_client()
    return _client

# Tavily搜索客户端
try:
    from tavily import TavilyClient
    
    if not config.tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found. Please set it in .env file or environment variables.")
    
    _tavily_client = TavilyClient(config.tavily_api_key)
except ImportError:
    logger.error("Tavily client not installed. Please install with 'pip install tavily-python'")
    raise

class SearchError(Exception):
    """自定义搜索错误类，用于重试逻辑"""
    pass

@retry(
    stop=stop_after_attempt(config.retry_attempts),
    wait=wait_exponential(multiplier=1, min=config.retry_min_wait, max=config.retry_max_wait),
    retry=retry_if_exception_type(SearchError),
    before_sleep=lambda retry_state: logger.warning(f"Search attempt {retry_state.attempt_number} failed, retrying...")
)
def web_search(query: str) -> List[Dict[str, Any]]:
    """
    封装tavily接口，返回list[dict]格式的检索结果
    
    Args:
        query: 搜索查询字符串
        
    Returns:
        检索结果列表
        
    Raises:
        SearchError: 当搜索失败时抛出
    """
    start_time = time.time()
    try:
        results = _tavily_client.search(
            query=query, 
            max_results=config.each_search_max_result
        )['results']
        
        logger.info(f"Search for '{query}' completed in {time.time() - start_time:.2f}s, found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise SearchError(f"Search failed: {str(e)}")

class Message(BaseModel):
    """消息模型"""
    role: str
    content: str
    prefix: Optional[bool] = None

def build_messages(query: str, role: str = 'user', history: Optional[List[Message]] = None) -> List[Message]:
    """
    构建消息列表
    
    Args:
        query: 当前消息内容
        role: 消息角色
        history: 历史消息
        
    Returns:
        构建好的消息列表
    """
    if role == 'assistant':
        last_one_message = {"role": role, "content": query, "prefix": True}
    else:
        last_one_message = {"role": role, "content": query}
    
    messages = []
    if history:
        messages.extend(history)
        messages.append(last_one_message)
    else:
        messages.append(last_one_message)
    
    return messages

def call_model(messages: List[Message], model_name: str = config.model_name) -> Dict[str, Any]:
    """
    调用模型API进行生成
    
    Args:
        messages: 消息列表
        model_name: 模型名称
        
    Returns:
        包含模型回复的字典
    """
    client = get_client()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            # frequency_penalty=config.frequency_penalty,
            # max_tokens = config.max_tokens
            stop=["<|im_end|>", "</tool_call>"]
        )
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        
        return {
            "content": content,
            "reasoning_content": reasoning_content
        }
    except Exception as e:
        logger.error(f"Model API call failed: {str(e)}")
        raise

def return_output(query: str, role: str = 'user', history: Optional[List[Message]] = None, model_name: str = config.model_name) -> Dict[str, Any]:
    """
    处理用户输入，调用模型，处理响应
    
    Args:
        query: 用户输入或当前消息
        role: 消息角色
        history: 历史消息
        model_name: 模型名称
        
    Returns:
        包含回答和历史的字典
    """
    if history:
        messages = build_messages(query, role=role, history=history)
    else:
        messages = build_messages(query, role=role)
    
    # 调用模型
    response = call_model(messages, model_name)
    reasoning_content = response["reasoning_content"]
    content = response["content"]
    
    # 处理工具调用
    if re.findall("<tool_call>(.*?)\s*$", content, re.DOTALL):
        content += '</tool_call>'
        messages.append({"role": "assistant", "content": content})
    # 未触发工具调用
    else:
        messages.append({"role": "assistant", "content": content})
    
    return {
        'question': query, 
        'answer': content, 
        'think': reasoning_content, 
        'history': messages
    }

# 工具调用匹配正则
TOOL_CALL_REGEX = r"<tool_call>(.*?)</tool_call>\s*$"

def non_stream_qa(
    user_question: str, 
    system_prompt: str = config.system_prompt, 
    history: Optional[List[Message]] = None, 
    max_search_times: int = config.max_search_times, 
    max_new_tokens: int = config.max_tokens
) -> Dict[str, Any]:
    """
    执行非流式问答，支持工具调用和多轮对话
    
    逻辑：
    1) 第一次调用模型generate，若生成中未出现<tool_call>, 则直接返回
    2) 若出现<tool_call>, 则解析它，执行web_search，拼<tool_response>，再次generate
    3) 重复执行，直到没有更多<tool_call>或达到最大搜索次数
    
    Args:
        user_question: 用户问题
        system_prompt: 系统提示词
        history: 历史消息
        max_search_times: 最大搜索次数
        max_new_tokens: 最大生成token数
        
    Returns:
        包含最终回答和思考过程的字典
    """
    logger.info(f"Processing question: {user_question}")
    start_time = time.time()
    
    think_list = []
    if not history:
        history = [{
            "role": "system",
            "content": system_prompt
        }]
    
    # 第一次调用模型的回答结果
    out = return_output(user_question, history=history)
    
    final_buffer = f"<think>\n{out['think']}\n</think>\n" + out['history'][-1]['content']
    think = out['think']
    think_list.append(think)
    new_buffer = out['history'][-1]['content']
    search_count = 0
    
    # 动态追踪最新用户消息
    user_indices = [i for i, msg in enumerate(out['history']) if msg.get('role') == "user"]
    user_max_index = user_indices[-1] if user_indices else 0

    # 工具调用循环
    while True:
        # 检查<tool_call>
        match = re.search(TOOL_CALL_REGEX, new_buffer, re.DOTALL)
        if match and search_count < max_search_times:
            search_count += 1
            tool_call_content = re.findall(TOOL_CALL_REGEX, new_buffer, re.DOTALL)[-1]
            
            try:
                tool_payload = json.loads(tool_call_content)
                
                if tool_payload["action"] == "web_search":
                    query_text = tool_payload["action_input"]["query"]
                    logger.info(f"Search #{search_count}: '{query_text}'")
                    
                    # 执行网络搜索
                    search_result = web_search(query_text)
                    
                    # 构造工具响应
                    tool_response = json.dumps({
                        "action": "web_search",
                        "action_input": {"query": query_text},
                        "response": search_result
                    }, ensure_ascii=False)
                    
                    tool_response_chunk = f"\n<tool_response>\n{tool_response}\n</tool_response>\n"
                    final_buffer += tool_response_chunk
                    
                    # 继续推理
                    out = return_output(
                        final_buffer, 
                        role='assistant', 
                        history=out['history'][:user_max_index+1]
                    )
                    
                    final_buffer += out['answer']
                    think_list.append(out['think'])
                    new_buffer = out['answer']      
                else:
                    # 未知工具
                    logger.warning(f"Unknown tool action: {tool_payload['action']}")
                    break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool payload: {e}")
                logger.error(f"Raw tool content: {tool_call_content}")
                break
        else:
            # 无更多<tool_call>或达到最大搜索次数
            if search_count >= max_search_times and re.search(TOOL_CALL_REGEX, new_buffer, re.DOTALL):
                logger.warning(f"Reached maximum search count ({max_search_times})")
            break

    # 整理输出
    out['history'] = out['history'][:user_max_index+1]
    out['history'].append({"role": "assistant", "content": final_buffer})
    
    processing_time = time.time() - start_time
    logger.info(f"Completed in {processing_time:.2f}s with {search_count} searches")
    
    return {
        'question': user_question, 
        'answer': final_buffer, 
        'first_think': think, 
        'history': out['history'], 
        "think_list": think_list,
        "search_count": search_count,
        "processing_time": processing_time
    }

def clean_chat_history_from_messages(messages: List[Message]) -> List[Message]:
    """
    从消息列表中清洗可读的聊天历史
    功能： 移除thinking+工具调用/响应标签的内容，简化+美化输出
    
    Args:
        messages: 消息列表
        
    Returns:
        清洗后的（格式化后的）聊天历史
    """
    chat_history = []
    for msg in messages:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            # 移除thinking+工具调用/响应标签，简化+美化输出（降低历史聊天对话的token数）
            content = re.sub(r'<think>.*?</think>\s*|<tool_call>.*?</tool_call>\s*|<tool_response>.*?</tool_response>\s*', '', content, flags=re.DOTALL)
            
            chat_history.append({
                'role': msg['role'],
                'content': content
            })
        else:
            chat_history.append({
                'role': msg['role'],
                'content': msg.get('content', '')
            })
    return chat_history

def main():
    """主程序入口"""
    # 单轮对话
    print("\n==== 单轮对话示例 ====")
    user_input = input('请输入你的问题：')
    final_out = non_stream_qa(user_input)
    print("\n全部回答（含思考+工具调用）：")
    print(final_out['answer'])
    
    # 移除thinking+工具调用/响应标签，美化输出
    clean_answer = re.sub(r'<think>.*?</think>\s*|<tool_call>.*?</tool_call>\s*|<tool_response>.*?</tool_response>\s*', '', final_out['answer'], flags=re.DOTALL)
    print("\n移除thinking和工具调用/响应后的回答：")
    print(clean_answer)

    # 多轮对话
    print("\n==== 多轮对话示例 ====")
    # history记录清洗，选择性使用 -> 是否清除移除thinking+工具调用/响应标签
    history = clean_chat_history_from_messages(final_out['history'])
    while True:
        user_input = input('\n请输入下一个问题 (输入 q 退出)：')
        if user_input.lower() == 'q':
            break
            
        final_out = non_stream_qa(user_input, history=history)
        print("\n全部回答（含思考+工具调用）：")
        print(final_out['answer'])
        history = clean_chat_history_from_messages(final_out['history'])
        
        # 移除thinking+工具调用/响应标签，美化输出
        clean_answer = re.sub(r'<think>.*?</think>\s*|<tool_call>.*?</tool_call>\s*|<tool_response>.*?</tool_response>\s*', '', final_out['answer'], flags=re.DOTALL)
        print("\n移除thinking和工具调用/响应后的回答：")
        print(clean_answer)
        
        # 显示搜索统计
        print(f"\n[统计] 使用了 {final_out['search_count']} 次搜索，处理时间: {final_out['processing_time']:.2f}秒")

if __name__ == "__main__":
    main()
