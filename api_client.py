"""
LLM Agent MCM - API客户端模块
支持OpenRouter API，并预留其他API格式兼容接口
"""

import aiohttp
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncGenerator
from config import APIProvider, APIConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """聊天消息"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """聊天响应"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


class BaseAPIClient(ABC):
    """API客户端基类"""

    def __init__(self, config: APIConfig):
        self.config = config

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        model: ModelConfig,
        **kwargs
    ) -> ChatResponse:
        """发送聊天请求"""
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Message],
        model: ModelConfig,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式聊天请求"""
        pass


class OpenRouterClient(BaseAPIClient):
    """OpenRouter API客户端"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mcm-agent",
            "X-Title": "MCM Agent"
        }

    async def chat(
        self,
        messages: List[Message],
        model: ModelConfig,
        max_retries: int = 3,
        **kwargs
    ) -> ChatResponse:
        """发送聊天请求"""

        payload = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", model.max_tokens),
            "temperature": kwargs.get("temperature", model.temperature),
        }

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.config.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return ChatResponse(
                                content=data["choices"][0]["message"]["content"],
                                model=data.get("model", model.model_id),
                                usage=data.get("usage", {}),
                                finish_reason=data["choices"][0].get("finish_reason", "stop")
                            )
                        elif response.status == 429:
                            # 速率限制，等待后重试
                            wait_time = 2 ** attempt * 5
                            logger.warning(f"Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            logger.error(f"API error {response.status}: {error_text}")
                            raise Exception(f"API error: {response.status} - {error_text}")

            except asyncio.TimeoutError:
                logger.warning(f"Request timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        raise Exception("Max retries exceeded")

    async def chat_stream(
        self,
        messages: List[Message],
        model: ModelConfig,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式聊天请求"""

        payload = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", model.max_tokens),
            "temperature": kwargs.get("temperature", model.temperature),
            "stream": True
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.base_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue


class OpenAIClient(BaseAPIClient):
    """OpenAI API客户端（兼容接口）"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: List[Message],
        model: ModelConfig,
        max_retries: int = 3,
        **kwargs
    ) -> ChatResponse:
        """发送聊天请求"""

        payload = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", model.max_tokens),
            "temperature": kwargs.get("temperature", model.temperature),
        }

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.config.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return ChatResponse(
                                content=data["choices"][0]["message"]["content"],
                                model=data.get("model", model.model_id),
                                usage=data.get("usage", {}),
                                finish_reason=data["choices"][0].get("finish_reason", "stop")
                            )
                        elif response.status == 429:
                            wait_time = 2 ** attempt * 5
                            logger.warning(f"Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error: {response.status} - {error_text}")
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        raise Exception("Max retries exceeded")

    async def chat_stream(
        self,
        messages: List[Message],
        model: ModelConfig,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式聊天请求"""
        # 与OpenRouter类似的实现
        payload = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", model.max_tokens),
            "temperature": kwargs.get("temperature", model.temperature),
            "stream": True
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.base_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue


class AnthropicClient(BaseAPIClient):
    """Anthropic API客户端（兼容接口）"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

    async def chat(
        self,
        messages: List[Message],
        model: ModelConfig,
        max_retries: int = 3,
        **kwargs
    ) -> ChatResponse:
        """发送聊天请求"""

        # 提取system消息
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": model.model_id,
            "max_tokens": kwargs.get("max_tokens", model.max_tokens),
            "messages": chat_messages,
        }
        if system_msg:
            payload["system"] = system_msg

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.config.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            content = ""
                            for block in data.get("content", []):
                                if block.get("type") == "text":
                                    content += block.get("text", "")
                            return ChatResponse(
                                content=content,
                                model=data.get("model", model.model_id),
                                usage=data.get("usage", {}),
                                finish_reason=data.get("stop_reason", "stop")
                            )
                        elif response.status == 429:
                            wait_time = 2 ** attempt * 5
                            logger.warning(f"Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error: {response.status} - {error_text}")
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        raise Exception("Max retries exceeded")

    async def chat_stream(
        self,
        messages: List[Message],
        model: ModelConfig,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式聊天请求"""
        # Anthropic流式API实现
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": model.model_id,
            "max_tokens": kwargs.get("max_tokens", model.max_tokens),
            "messages": chat_messages,
            "stream": True
        }
        if system_msg:
            payload["system"] = system_msg

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.base_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue


class InternClient(BaseAPIClient):
    """书生·浦语 Intern-S1 API客户端

    兼容OpenAI格式的API，支持思考模式(thinking_mode)
    API文档: https://internlm.intern-ai.org.cn/api/tokens
    """

    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: List[Message],
        model: ModelConfig,
        max_retries: int = 3,
        **kwargs
    ) -> ChatResponse:
        """发送聊天请求"""

        payload = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", model.temperature),
        }

        # Intern API 不需要显式设置 max_tokens，使用默认值
        if kwargs.get("max_tokens"):
            payload["max_tokens"] = kwargs.get("max_tokens")

        # 支持思考模式
        if kwargs.get("thinking_mode", False):
            payload["extra_body"] = {"thinking_mode": True}

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.config.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return ChatResponse(
                                content=data["choices"][0]["message"]["content"],
                                model=data.get("model", model.model_id),
                                usage=data.get("usage", {}),
                                finish_reason=data["choices"][0].get("finish_reason", "stop")
                            )
                        elif response.status == 429:
                            wait_time = 2 ** attempt * 5
                            logger.warning(f"Intern API rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            logger.error(f"Intern API error {response.status}: {error_text}")
                            raise Exception(f"Intern API error: {response.status} - {error_text}")

            except asyncio.TimeoutError:
                logger.warning(f"Intern API timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        raise Exception("Intern API max retries exceeded")

    async def chat_stream(
        self,
        messages: List[Message],
        model: ModelConfig,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式聊天请求"""

        payload = {
            "model": model.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", model.temperature),
            "stream": True
        }

        if kwargs.get("max_tokens"):
            payload["max_tokens"] = kwargs.get("max_tokens")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.base_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Intern API error: {response.status} - {error_text}")

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue


def create_client(config: APIConfig) -> BaseAPIClient:
    """工厂函数：根据配置创建对应的API客户端"""
    clients = {
        APIProvider.OPENROUTER: OpenRouterClient,
        APIProvider.OPENAI: OpenAIClient,
        APIProvider.ANTHROPIC: AnthropicClient,
        APIProvider.INTERN: InternClient,
    }

    client_class = clients.get(config.provider)
    if client_class is None:
        raise ValueError(f"Unsupported API provider: {config.provider}")

    return client_class(config)


class LLMAgent:
    """LLM Agent封装类"""

    def __init__(self, config: APIConfig, model: ModelConfig):
        self.client = create_client(config)
        self.model = model
        self.conversation_history: List[Message] = []

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        # 移除已有的system消息
        self.conversation_history = [
            m for m in self.conversation_history if m.role != "system"
        ]
        # 添加新的system消息
        self.conversation_history.insert(0, Message(role="system", content=prompt))

    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.conversation_history.append(Message(role=role, content=content))

    def clear_history(self, keep_system: bool = True):
        """清除对话历史"""
        if keep_system:
            self.conversation_history = [
                m for m in self.conversation_history if m.role == "system"
            ]
        else:
            self.conversation_history = []

    async def chat(self, user_message: str, **kwargs) -> str:
        """发送消息并获取回复"""
        self.add_message("user", user_message)

        response = await self.client.chat(
            messages=self.conversation_history,
            model=self.model,
            **kwargs
        )

        self.add_message("assistant", response.content)
        return response.content

    async def chat_stream(self, user_message: str, **kwargs) -> AsyncGenerator[str, None]:
        """流式发送消息并获取回复"""
        self.add_message("user", user_message)

        full_response = ""
        async for chunk in self.client.chat_stream(
            messages=self.conversation_history,
            model=self.model,
            **kwargs
        ):
            full_response += chunk
            yield chunk

        self.add_message("assistant", full_response)

    async def single_query(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> str:
        """单次查询（不保存历史）"""
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message)
        ]

        response = await self.client.chat(
            messages=messages,
            model=self.model,
            **kwargs
        )

        return response.content
