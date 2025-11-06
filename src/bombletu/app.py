"""应用主循环与入口。"""

from __future__ import annotations

import asyncio
import os
import signal
from time import time

from asyncio_channel import create_channel, create_sliding_buffer
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from ncatbot.core import BotClient
from ncatbot.core.event import GroupMessageEvent

from .config import CON, GRP, USR, get_log
from .graph import make_agent
from .msgfmt import msglfmt

logger = get_log(__name__)


class App:
    """封装 BotClient 与消息通道的应用容器。"""

    def __init__(self) -> None:
        async def group_message_handler(event: GroupMessageEvent):
            if event.group_id == GRP:
                await self.newmsgchan.put(event)
                if event.message.is_user_at(USR):
                    logger.info(
                        "group_message_handler: 提及我的消息 %s 送入 intrchan",
                        event.raw_message,
                    )
                    await self.intrchan.put("提及我的消息")
            elif event.group_id == CON:
                if event.raw_message == "/kill":
                    logger.warning("panic called")
                    os.kill(os.getpid(), signal.SIGKILL)
                elif event.raw_message == "/term":
                    logger.info("exit called")
                    os.kill(os.getpid(), signal.SIGTERM)
                elif event.raw_message == "/int":
                    logger.info("exit called")
                    os.kill(os.getpid(), signal.SIGINT)

        qbot = BotClient()
        qbot.add_group_message_handler(group_message_handler)  # type: ignore[arg-type]
        qbot.add_startup_handler(make_agent_loop(self))  # type: ignore[arg-type]

        msgchan = create_channel(create_sliding_buffer(100))  # type: ignore[arg-type]
        intrchan = create_channel(create_sliding_buffer(1))  # type: ignore[arg-type]

        self.qbot = qbot
        self.newmsgchan = msgchan
        self.intrchan = intrchan
        self.unread: list[GroupMessageEvent] = []

    async def wait_intr(self, minutes: int):
        target_time = time() + minutes * 60
        while True:
            intr = await self.intrchan.take(timeout=5)
            if intr is not None:
                return intr
            if time() > target_time:
                return None

    async def collect_unread(self) -> int:
        collected = 0
        while ev := await self.newmsgchan.take(timeout=0):
            self.unread.append(ev)
            collected += 1
        return collected

    async def get_unread(self, limit: int) -> str:
        await self.collect_unread()
        events: list[GroupMessageEvent] = []
        for _ in range(limit):
            if not self.unread:
                break
            events.append(self.unread.pop(0))
        summary = msglfmt(events) if events else ""
        parts = [summary] if summary else []
        parts.append(f"[unread {len(self.unread)}]")
        return "\n".join(parts)

    def run(self) -> None:
        self.qbot.run_frontend()


def check_idle_call(invocation):
    if (
        "messages" in invocation
        and isinstance(invocation["messages"], list)
        and invocation["messages"]
        and isinstance(last_msg := invocation["messages"][-1], AIMessage)
        and len(last_msg.tool_calls) == 1
        and (call := last_msg.tool_calls[0])["name"] == "idle"
    ):
        call_id = call["id"]
        minutes = int(call["args"]["minutes"])
        return call_id, minutes
    return None, 0


async def agent_loop(app: App):
    logger.info("agent loop starting")
    agent = make_agent()
    agentconfig = RunnableConfig(configurable={"thread_id": 1, "app": app, "qapi": app.qbot.api})
    msg_inject: list[HumanMessage | ToolMessage] = []
    while True:
        logger.info("agent invoking")
        ret = await agent.ainvoke({"messages": msg_inject}, config=agentconfig, print_mode="updates")  # type: ignore[arg-type]
        logger.debug("agent return: %s", ret)
        idle_id, minutes = check_idle_call(ret)
        if idle_id:
            logger.info("agent sleeping %s min", minutes)
        else:
            logger.info("agent continuing")
        intr = await app.wait_intr(minutes)
        unread = await app.collect_unread()
        msg_inject = []
        if idle_id is not None:
            if intr:
                msg_inject.append(ToolMessage("Idle interrupted!", tool_call_id=idle_id))
            else:
                msg_inject.append(ToolMessage("Idle finished.", tool_call_id=idle_id))
        info_inject: list[str] = []
        if intr:
            info_inject.append(f"[notify {intr}]")
        if unread > 0:
            info_inject.append(f"[event {unread}条新消息]")
        if info_inject:
            msg_inject.append(HumanMessage("\n".join(info_inject)))


def make_agent_loop(app: App):
    async def agent_loop_wrapper(_):
        while True:
            await asyncio.sleep(10)
            try:
                await agent_loop(app)
            except BaseException as exc:  # noqa: BLE001
                logger.error("agent loop exception: %s", exc)
                await app.qbot.api.send_group_text(
                    GRP,
                    "Someone tell [CQ:at,qq=1571224208] there is a problem with my AI.",
                )
                await app.qbot.api.send_group_text(CON, f"{GRP} {exc}")

    return agent_loop_wrapper


def run() -> None:
    App().run()


__all__ = ["App", "agent_loop", "make_agent_loop", "run"]
