"""LangChain 工具定义。"""

import subprocess
from datetime import datetime
from os import environ

import pytz
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from ncatbot.core.api import BotAPI, NapCatAPIError

from .config import GRP, TZ, get_log
from .msgfmt import msglfmt, parse_msg

logger = get_log(__name__)


def get_api(cfg: RunnableConfig) -> BotAPI:
    return cfg["configurable"]["qapi"]  # type: ignore[index]


@tool
def date() -> str:
    """获取当前的本地日期和时间。"""

    return datetime.now(pytz.timezone(TZ)).isoformat(timespec="seconds")


@tool
async def send(cfg: RunnableConfig, content: str) -> str:
    """在群里发送消息。"""

    logger.info("send: %s", content)
    try:
        await get_api(cfg).send_group_msg(GRP, parse_msg(content).to_list())
        return "[success]"
    except NapCatAPIError as exc:
        logger.warning("get_message error: %s", exc)
        return "[error 软件暂时故障]"


@tool
async def get_unread(cfg: RunnableConfig, limit: int) -> str:
    """获取未读消息列表。"""

    app = cfg["configurable"]["app"]  # type: ignore[index]
    return await app.get_unread(limit)


@tool
async def get_messages(cfg: RunnableConfig, fro: int, to: int, with_id: bool = False) -> str:
    """查阅消息记录。"""

    logger.info("get_messages: %s, %s, %s", fro, to, with_id)
    try:
        history = await get_api(cfg).get_group_msg_history(GRP, 0, fro)
        return msglfmt(history[: fro - to + 1], with_id)
    except NapCatAPIError as exc:
        logger.warning("get_message error: %s", exc)
        return "[error 软件暂时故障]"


@tool
async def get_messages_by_id(
    cfg: RunnableConfig, id: str, before: int = 0, after: int = 0
) -> str:
    """按消息 ID 查阅消息记录。"""

    api = get_api(cfg)
    try:
        before_msgs = await api.get_group_msg_history(GRP, id, before + 1, True)
        after_msgs = await api.get_group_msg_history(GRP, id, after + 1, False)
        messages = before_msgs + after_msgs[1:]
        return msglfmt(messages, id)
    except NapCatAPIError as exc:
        logger.error("%s", exc)
        return "[error 软件暂时故障]"


visual_model = ChatOpenAI(
    temperature=0.6,
    model=environ["VIS_MODEL"],
    api_key=environ["VIS_API_KEY"],  # type:ignore[index]
    base_url=environ["VIS_BASE_URL"],
)


@tool
async def ask_image(
    cfg: RunnableConfig, file_name: str, prompt: str = "群友发了这个图，什么意思？"
) -> str:
    """向视觉模型询问图像问题。"""

    try:
        img = await get_api(cfg).get_image(file=file_name)
    except BaseException as exc:  # noqa: BLE001
        logger.error("ask_image 图像获取失败 %s", exc)
        return "[error 图像打开失败]"
    fpath = img.file.replace("/app/.config/QQ/", "./data/napcat/config_qq/")
    b64img = subprocess.run(
        ["sudo", "base64", fpath], capture_output=True, text=True, check=False
    ).stdout.strip()
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64img}"},
            },
        ]
    )
    try:
        ret = await visual_model.ainvoke([message])
    except BaseException as exc:  # noqa: BLE001
        logger.error("模型调用出错 %s", exc)
        return "[error 模型调用出错]"
    return str(ret.content)


ALL_TOOLS = [
    date,
    send,
    get_unread,
    get_messages,
    get_messages_by_id,
    ask_image,
]

__all__ = [
    "ALL_TOOLS",
    "ask_image",
    "date",
    "get_messages",
    "get_messages_by_id",
    "get_unread",
    "send",
]
