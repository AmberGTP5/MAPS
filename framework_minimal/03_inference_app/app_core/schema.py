from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ModelResponse(BaseModel):
    """标准化的模型返回结果"""
    content: Optional[str] = ""
    tool_calls: Optional[List[ToolCall]] = []
    message: Message  # 原始消息对象，用于存入历史