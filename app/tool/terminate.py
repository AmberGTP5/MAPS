from app.tool.base import BaseTool

_TERMINATE_DESCRIPTION = """Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work."""

class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: str):
        """Finish the current execution"""
        # 使用 success_response 包装返回值
        # 这样返回的是 ToolResult 对象，而不是 str
        return self.success_response(f"The interaction has been completed with status: {status}")