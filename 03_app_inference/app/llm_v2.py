from openai import AsyncOpenAI
from app.config import config
from app.schema import Message, ModelResponse, ToolCall, FunctionCall
from app.logger import logger

class LLM:
    def __init__(self, config_key="default"):
        # 根据传入的 key 读取不同的配置
        if config_key not in config.llm:
            logger.warning(f"Config key '{config_key}' not found, falling back to 'default'")
            config_key = "default"
            
        llm_config = config.llm[config_key]
        
        self.client = AsyncOpenAI(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/FoundationAgents/OpenManus",
                "X-Title": "OpenManus-Robot"
            }
        )
        self.model = llm_config.model
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature

        logger.info(f"Initialized LLM [{config_key}] with model: {self.model}")

    async def chat(
        self, 
        messages: list[dict], 
        tools: list[dict] = None
    ) -> ModelResponse:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            # logger.info(f"Sending request to {self.model}...") # 可以注释掉避免刷屏
            response = await self.client.chat.completions.create(**params)
            
            choice = response.choices[0]
            msg = choice.message
            
            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        function=FunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments
                        )
                    ))

            return ModelResponse(
                content=msg.content,
                tool_calls=tool_calls,
                message=Message(
                    role="assistant",
                    content=msg.content,
                    tool_calls=tool_calls if tool_calls else None
                )
            )

        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            raise e