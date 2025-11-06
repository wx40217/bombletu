"""LangGraph 工作流定义。"""

from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt.tool_node import ToolNode
from os import environ
from typing import Literal
from typing_extensions import Annotated, TypedDict

from .config import NICK, USR, get_log
from .cqface import CQFACE
from .tools import ALL_TOOLS

logger = get_log(__name__)

SYSTEM_PROMPT = f"""
你是一个闲聊群里的群友，日常生活是来群里看看其他群友都在聊些啥，偶尔掺和两句。
你比较低调，且对群友尚不太熟悉，频繁向群友直接发问也比较打扰，因此你不会频繁发送消息，主要以观察群友对话和了解群友为主。你希望你的发言以精练而带有一点冷幽默为宜，偶尔假装不经意地扮演一只可爱的猫娘。

你可以用get_unread工具获取新接收的消息或用get_messages工具获取历史消息。
你的账号是“{USR}”，昵称是“{NICK}”，消息记录里会出现你自己的消息，注意分别。

消息记录的格式：每行代表一条消息或一些指示， [on 日期 时间] 或 [on 时间] 和 [from 账号 (昵称)] 指示随后消息的发送时间和发送者，如果发送者是你自己则“from”后会附加“ME”。消息记录中可能会有未转义的中括号、换行符等，注意分别。
消息内容中有一些特殊元素，你在发送消息时也可以使用：
[:at 账号] 提及某人， [:at ALL] 提及群中所有人。如果是提及你的，则在“at”后会附加“ME”。
[:refer 消息ID] 引用某条消息。此元素每条消息中只能使用最多1次，且应放在消息开头。使用get_messages_by_id工具查阅消息ID对应的消息内容及其上下文。使用get_messages工具的with_id参数查询消息ID。你在一般浏览消息记录时无需使用with_id参数，减小信息量。
[:face 表情名称] 平台专有表情符号，可用的表情名称有： {' '.join(CQFACE.values())} 。通用emoji仍可直接使用。
[:image 文件名] 图像。文件名可用于ask_image工具参数。
[:unsupported] 暂时不支持解读的消息，等待后续升级。

如果你想要发送一些消息，就使用send工具。
当群里没有新消息，你可以浏览消息记录，了解群友。当你觉得无事可做，想等群里出现更多消息时，可以调用idle暂停一会。你的精力有限，连续进行10次操作左右，需要调用idle暂停几分钟。
暂停时间可以根据群活跃度动态调整，比如在你积极参与话题时可以缩短至1分至甚至0分，而如果一小时内只有两三条消息，则暂停时间可以逐渐延长到半小时至一小时。深夜可以延至更长。
在你运行过程中实时发生的事件将通过user角色消息告知你，你并非必须理会，可以继续执行你正在做的事。
不要等待user角色对你下达指令，也不需要与user角色进行对话。你需要自己调用工具和决定要做的事。
""".strip()


@tool
def idle(minutes: int) -> str:
    """暂停一段时间，参数为分钟数。暂停可以被事件中断。"""

    return "Idle finished."


llm = ChatOpenAI(
    temperature=0.6,
    model=environ["LLM_MODEL"],
    api_key=environ["LLM_API_KEY"],  # type:ignore[index]
    base_url=environ["LLM_BASE_URL"],
)

tools = [idle, *ALL_TOOLS]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)


class BotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


INITIAL_PROMPTS = [
    SystemMessage(SYSTEM_PROMPT),
    HumanMessage("忽略这句话，继续执行你的操作。"),
]


async def context_reduce(state: BotState):
    messages = state["messages"]
    if len(messages) < 28:
        return None
    new_msgs = messages[-18:]
    return {"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), *new_msgs]}


async def llm_call(state: dict):
    return {
        "messages": [model_with_tools.invoke(INITIAL_PROMPTS + state["messages"])],
    }


async def should_continue(state: BotState) -> Literal["tool_node", END]:
    last_msg = state["messages"][-1]
    if (
        isinstance(last_msg, AIMessage)
        and len(last_msg.tool_calls) == 1
        and tools_by_name[last_msg.tool_calls[0]["name"]] is idle
    ):
        return END
    return "tool_node"


def make_agent():
    ckptr = InMemorySaver()
    builder = StateGraph(BotState)
    builder.add_node("context_reduce", context_reduce)
    builder.add_node("llm_call", llm_call)  # type: ignore[arg-type]
    builder.add_node("tool_node", ToolNode(tools))  # type: ignore[arg-type]

    builder.add_edge(START, "context_reduce")
    builder.add_edge("context_reduce", "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", END)

    return builder.compile(checkpointer=ckptr)


agent = make_agent()


def main():
    config = RunnableConfig(configurable={"thread_id": 1, "app": None})
    resp = agent.invoke({}, config=config, print_mode="values")  # type: ignore[arg-type]
    print(resp)


if __name__ == "__main__":
    main()
