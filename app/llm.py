import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from app.config import config
from app.logger import logger
from app.schema import FunctionCall, Message, ModelResponse, ToolCall


@dataclass
class LLMRuntimeConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    source: str
    local_mode: bool = False
    local_model_path: str = ""
    local_adapter_path: str = ""
    local_dtype: str = "bfloat16"
    local_device_map: str = "auto"


_LOCAL_BACKEND_CACHE: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}
_LOCAL_BACKEND_LOCK = threading.Lock()


def _env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _resolve_runtime_config(config_key: str) -> LLMRuntimeConfig:
    if config_key not in config.llm:
        logger.warning(f"Config key '{config_key}' not found, falling back to 'default'")
        config_key = "default"

    llm_config = config.llm[config_key]
    role = config_key.upper()

    base_url = llm_config.base_url
    api_key = llm_config.api_key
    model = llm_config.model
    max_tokens = llm_config.max_tokens
    temperature = llm_config.temperature
    source = "config.toml"
    local_mode = False
    local_model_path = ""
    local_adapter_path = ""
    local_dtype = _env(f"ROBOT_{role}_LOCAL_DTYPE", "ROBOT_LOCAL_DTYPE") or "bfloat16"
    local_device_map = _env(f"ROBOT_{role}_LOCAL_DEVICE_MAP", "ROBOT_LOCAL_DEVICE_MAP") or "auto"

    # Highest priority: explicit runtime override for local/remote OpenAI-compatible endpoints.
    runtime_base_url = _env(f"ROBOT_{role}_BASE_URL", "ROBOT_BASE_URL")
    runtime_api_key = _env(f"ROBOT_{role}_API_KEY", "ROBOT_API_KEY")
    runtime_model = _env(f"ROBOT_{role}_MODEL", "ROBOT_MODEL")
    runtime_max_tokens = _env(f"ROBOT_{role}_MAX_TOKENS", "ROBOT_MAX_TOKENS")
    runtime_temperature = _env(f"ROBOT_{role}_TEMPERATURE", "ROBOT_TEMPERATURE")
    runtime_local_model = _env(f"ROBOT_{role}_LOCAL_MODEL_PATH", "ROBOT_LOCAL_MODEL_PATH")
    runtime_local_adapter = _env(f"ROBOT_{role}_LOCAL_ADAPTER_PATH", "ROBOT_LOCAL_ADAPTER_PATH")

    if runtime_local_model:
        local_mode = True
        local_model_path = runtime_local_model
        local_adapter_path = runtime_local_adapter
        model = _env(f"ROBOT_{role}_LOCAL_MODEL_NAME", "ROBOT_LOCAL_MODEL_NAME") or model
        max_tokens = _as_int(runtime_max_tokens, max_tokens)
        temperature = _as_float(runtime_temperature, temperature)
        source = "ROBOT_* local hf"

    elif runtime_base_url or runtime_api_key or runtime_model or runtime_max_tokens or runtime_temperature:
        base_url = runtime_base_url or base_url
        api_key = runtime_api_key or api_key
        model = runtime_model or model
        max_tokens = _as_int(runtime_max_tokens, max_tokens)
        temperature = _as_float(runtime_temperature, temperature)
        source = "ROBOT_* env"
    else:
        # Legacy OpenRouter overrides are kept for backward compatibility.
        openrouter_api_key = _env("OPENROUTER_API_KEY")
        if openrouter_api_key:
            base_url = _env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
            api_key = openrouter_api_key
            model = _env(f"OPENROUTER_{role}_MODEL") or _env("OPENROUTER_MODEL") or model
            max_tokens = _as_int(_env("OPENROUTER_MAX_TOKENS"), max_tokens)
            temperature = _as_float(_env("OPENROUTER_TEMPERATURE"), temperature)
            source = "OPENROUTER_* env"

    # OpenAI SDK requires a non-empty key; local OpenAI-compatible servers usually accept any string.
    if not api_key:
        api_key = "EMPTY"

    return LLMRuntimeConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        source=source,
        local_mode=local_mode,
        local_model_path=local_model_path,
        local_adapter_path=local_adapter_path,
        local_dtype=local_dtype,
        local_device_map=local_device_map,
    )


class LLM:
    def __init__(self, config_key: str = "default"):
        runtime = _resolve_runtime_config(config_key)
        self.local_mode = runtime.local_mode

        self.client = None
        self.local_tokenizer = None
        self.local_model = None

        if self.local_mode:
            self.local_tokenizer, self.local_model = self._load_local_backend(
                model_path=runtime.local_model_path,
                adapter_path=runtime.local_adapter_path,
                dtype=runtime.local_dtype,
                device_map=runtime.local_device_map,
            )
        else:
            self.client = AsyncOpenAI(
                base_url=runtime.base_url,
                api_key=runtime.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/FoundationAgents/OpenManus",
                    "X-Title": "OpenManus-Robot",
                },
            )
        self.model = runtime.model
        self.max_tokens = runtime.max_tokens
        self.temperature = runtime.temperature

        logger.info(
            f"Initialized LLM [{config_key}] from {runtime.source} with model={self.model}, base_url={runtime.base_url}"
        )

    @staticmethod
    def _load_local_backend(model_path: str, adapter_path: str, dtype: str, device_map: str) -> tuple[Any, Any]:
        cache_key = (model_path, adapter_path, dtype, device_map)
        with _LOCAL_BACKEND_LOCK:
            cached = _LOCAL_BACKEND_CACHE.get(cache_key)
            if cached is not None:
                return cached

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype.lower(), torch.bfloat16)

        logger.info(
            f"Loading local HF backend: model_path={model_path}, adapter_path={adapter_path or '-'}, "
            f"dtype={dtype}, device_map={device_map}"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        if adapter_path:
            from peft import PeftModel

            logger.info(f"Loading local PEFT adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

        model.eval()
        with _LOCAL_BACKEND_LOCK:
            _LOCAL_BACKEND_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        if self.local_tokenizer is not None and hasattr(self.local_tokenizer, "apply_chat_template"):
            return self.local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        chunks = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip()
            content = str(msg.get("content", "")).strip()
            chunks.append(f"{role}: {content}")
        chunks.append("assistant:")
        return "\n".join(chunks)

    def _local_generate_sync(self, messages: list[dict]) -> str:
        import torch

        prompt = self._messages_to_prompt(messages)
        assert self.local_tokenizer is not None
        assert self.local_model is not None

        inputs = self.local_tokenizer(prompt, return_tensors="pt")
        first_param = next(self.local_model.parameters())
        inputs = {k: v.to(first_param.device) for k, v in inputs.items()}

        do_sample = self.temperature is not None and self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": int(self.max_tokens),
            "do_sample": bool(do_sample),
            "pad_token_id": self.local_tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(self.temperature)
            gen_kwargs["top_p"] = 0.95

        with torch.no_grad():
            output_ids = self.local_model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0][prompt_len:]
        return self.local_tokenizer.decode(gen_ids, skip_special_tokens=True)

    async def _local_generate(self, messages: list[dict]) -> str:
        return await asyncio.to_thread(self._local_generate_sync, messages)

    async def chat_stream(self, messages, tools=None):
        if self.local_mode:
            content = await self._local_generate(messages)
            chunk_size = 96
            for idx in range(0, len(content), chunk_size):
                yield content[idx: idx + chunk_size]
            return

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.error(f"Stream Error: {exc}")
            raise

    async def chat(self, messages: list[dict], tools: list[dict] = None) -> ModelResponse:
        if self.local_mode:
            content = await self._local_generate(messages)
            return ModelResponse(
                content=content,
                tool_calls=[],
                message=Message(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                ),
            )

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
            response = await self.client.chat.completions.create(**params)
            choice = response.choices[0]
            msg = choice.message

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            function=FunctionCall(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                    )

            return ModelResponse(
                content=msg.content,
                tool_calls=tool_calls,
                message=Message(
                    role="assistant",
                    content=msg.content,
                    tool_calls=tool_calls if tool_calls else None,
                ),
            )
        except Exception as exc:
            logger.error(f"LLM Error: {exc}")
            raise
