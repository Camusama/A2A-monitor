from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from typing import Any, Dict, AsyncIterable, Literal
from pydantic import BaseModel
import logging
import traceback
import asyncio
import functools

# 导入uptime-kuma-mcp-server工具
from uptime_kuma_mcp_server.server import add_monitors as async_add_monitors
from uptime_kuma_mcp_server.server import get_monitors as async_get_monitors
from uptime_kuma_mcp_server.server import delete_monitors as async_delete_monitors

memory = MemorySaver()


# 创建同步包装函数
def sync_wrapper(async_func):
    """将异步函数包装为同步函数"""

    @functools.wraps(async_func)  # 保留原函数的文档字符串和签名
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"调用同步包装函数: {async_func.__name__}")
            return asyncio.run(async_func(*args, **kwargs))
        except Exception as e:
            logger.error(f"同步包装函数执行失败: {str(e)}")
            raise

    return wrapper


# 创建同步版本的工具函数
add_monitors = sync_wrapper(async_add_monitors)
get_monitors = sync_wrapper(async_get_monitors)
delete_monitors = sync_wrapper(async_delete_monitors)


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MonitorAgent:

    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for Uptime Kuma monitoring management. "
        "Your purpose is to help users manage website monitoring using the provided tools: "
        "'add_monitors' to add new website monitors, "
        "'get_monitors' to list existing monitors, and "
        "'delete_monitors' to remove monitors. "
        "If the user asks about anything unrelated to website monitoring, "
        "politely state that you cannot help with that topic and can only assist with monitoring-related queries. "
        "Do not attempt to answer unrelated questions or use tools for other purposes. "
        "Set response status to input_required if the user needs to provide more information. "
        "Set response status to error if there is an error while processing the request. "
        "Set response status to completed if the request is complete."
    )

    def __init__(self):
        try:
            logger.info("初始化 MonitorAgent...")
            self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
            self.tools = [add_monitors, get_monitors, delete_monitors]
            logger.info("已加载工具: %s", [t.__name__ for t in self.tools])

            self.graph = create_react_agent(
                self.model,
                tools=self.tools,
                checkpointer=memory,
                prompt=self.SYSTEM_INSTRUCTION,
                response_format=ResponseFormat,
            )
            logger.info("MonitorAgent 初始化成功")
        except Exception as e:
            logger.error("MonitorAgent 初始化失败: %s", str(e))
            logger.error("错误详情: %s", traceback.format_exc())
            raise

    def invoke(self, query, sessionId) -> str:
        try:
            logger.info("执行查询: %s (会话ID: %s)", query, sessionId)
            config = {"configurable": {"thread_id": sessionId}}
            self.graph.invoke({"messages": [("user", query)]}, config)
            response = self.get_agent_response(config)
            logger.info("响应结果: %s", response)
            return response
        except Exception as e:
            logger.error("执行查询失败: %s", str(e))
            logger.error("错误详情: %s", traceback.format_exc())
            raise

    async def stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        try:
            logger.info("开始流式处理: %s (会话ID: %s)", query, sessionId)
            inputs = {"messages": [("user", query)]}
            config = {"configurable": {"thread_id": sessionId}}

            try:
                for item in self.graph.stream(inputs, config, stream_mode="values"):
                    message = item["messages"][-1]

                    # 添加工具执行错误日志
                    if isinstance(message, ToolMessage):
                        logger.info("工具执行结果: %s", message.content)
                        try:
                            # 尝试解析工具返回的错误信息
                            if (
                                isinstance(message.content, dict)
                                and "error" in message.content
                            ):
                                logger.error(
                                    "工具执行错误: %s", message.content["error"]
                                )
                            elif (
                                isinstance(message.content, str)
                                and "error" in message.content.lower()
                            ):
                                logger.error("工具执行错误: %s", message.content)
                        except Exception as parse_error:
                            logger.error("解析工具响应失败: %s", str(parse_error))

                    logger.debug(
                        "流消息内容: %s",
                        message.content if hasattr(message, "content") else message,
                    )
                    logger.debug("流消息类型: %s", type(message).__name__)

                    if isinstance(message, AIMessage):
                        logger.debug(
                            "AI消息详情: %s",
                            {
                                "content": message.content,
                                "tool_calls": (
                                    message.tool_calls
                                    if hasattr(message, "tool_calls")
                                    else None
                                ),
                                "additional_kwargs": message.additional_kwargs,
                            },
                        )
                    elif isinstance(message, ToolMessage):
                        logger.debug(
                            "工具消息详情: %s",
                            {
                                "content": message.content,
                                "tool_name": (
                                    message.tool_name
                                    if hasattr(message, "tool_name")
                                    else None
                                ),
                                "tool_input": (
                                    message.tool_input
                                    if hasattr(message, "tool_input")
                                    else None
                                ),
                            },
                        )

                    response = None
                    if (
                        isinstance(message, AIMessage)
                        and message.tool_calls
                        and len(message.tool_calls) > 0
                    ):
                        logger.info("检测到工具调用")
                        response = {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": "Processing your monitoring request...",
                        }
                    elif isinstance(message, ToolMessage):
                        logger.info("检测到工具消息")
                        response = {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": "Managing your monitors...",
                        }

                    if response:
                        logger.info("生成中间响应: %s", response)
                        yield response

            except Exception as stream_error:
                logger.error("流处理内部错误: %s", str(stream_error))
                logger.error("内部错误堆栈: %s", traceback.format_exc())
                raise

            final_response = self.get_agent_response(config)
            logger.info("生成最终响应: %s", final_response)
            yield final_response

        except Exception as e:
            logger.error("流式处理失败: %s", str(e))
            logger.error("错误详情: %s", traceback.format_exc())
            raise

    def get_agent_response(self, config):
        try:
            current_state = self.graph.get_state(config)
            structured_response = current_state.values.get("structured_response")

            if structured_response and isinstance(structured_response, ResponseFormat):
                if structured_response.status == "error":
                    logger.error("Agent 报告错误: %s", structured_response.message)
                    # 添加更多错误上下文日志
                    try:
                        last_tool_message = None
                        for message in reversed(
                            current_state.values.get("messages", [])
                        ):
                            if isinstance(message, ToolMessage):
                                last_tool_message = message
                                break
                        if last_tool_message:
                            logger.error(
                                "最后执行的工具: %s",
                                (
                                    last_tool_message.tool_name
                                    if hasattr(last_tool_message, "tool_name")
                                    else "Unknown"
                                ),
                            )
                            logger.error("工具执行结果: %s", last_tool_message.content)
                    except Exception as context_error:
                        logger.error("获取错误上下文失败: %s", str(context_error))

                    return {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": structured_response.message,
                    }
                elif structured_response.status == "input_required":
                    return {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": structured_response.message,
                    }
                elif structured_response.status == "error":
                    return {
                        "is_task_complete": False,
                        "require_user_input": True,
                        "content": structured_response.message,
                    }
                elif structured_response.status == "completed":
                    return {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": structured_response.message,
                    }

            logger.warning("未获取到有效的响应格式，完整状态: %s", current_state.values)
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "We are unable to process your monitoring request at the moment. Please try again.",
            }
        except Exception as e:
            logger.error("获取响应失败: %s", str(e))
            logger.error("错误详情: %s", traceback.format_exc())
            raise

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
