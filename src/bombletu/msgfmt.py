"""消息格式化与解析工具。"""

import re
from datetime import datetime
from typing import List

from ncatbot.core import MessageArray
from ncatbot.core.event import GroupMessageEvent
from ncatbot.core.event.message_segment.message_segment import (
    At,
    AtAll,
    Face,
    Image,
    PlainText,
    Reply,
)
from pytz import timezone

from .config import TZ, USR, get_log
from .cqface import CQFACE, RCQFACE

logger = get_log(__name__)

MSGP = re.compile(r"(\[\:.*?\])")


def format_face(face_id: str) -> str:
    if not face_id.isdigit():
        logger.warning("invalid face id %s", face_id)
        return face_id
    iid = int(face_id)
    if iid not in CQFACE:
        logger.warning("unknown face id %s", face_id)
        return face_id
    return CQFACE[iid]


def parse_face(name: str) -> str:
    if name not in RCQFACE:
        logger.warning("unknown face name %s", name)
        return "10068"  # 问号
    return RCQFACE[name]


def parse_msg(msg: str) -> MessageArray:
    segments = re.split(MSGP, msg)
    message = MessageArray()
    for seg in segments:
        if not (seg.startswith("[:") and seg.endswith("]")):
            message += PlainText(seg)
            continue
        sq = seg[2:-1]
        parts = sq.split()
        cmd = parts[0]
        if cmd == "at":
            if parts[1] == "ALL":
                message += AtAll()
            else:
                message += At(parts[1])
        elif cmd == "face":
            message += Face(parse_face(parts[1]))
        elif cmd == "refer":
            message += Reply(parts[1])
        else:
            logger.warning("unknown seg: %s", seg)
            message += PlainText(seg)
    return message


def format_msg(msg: MessageArray) -> str:
    content = ""
    for seg in msg:
        if isinstance(seg, PlainText):
            content += seg.text
        elif isinstance(seg, AtAll):
            content += "[:at ALL]"
        elif isinstance(seg, At):
            content += f"[:at ME {seg.qq}]" if seg.qq == USR else f"[:at {seg.qq}]"
        elif isinstance(seg, Face):
            content += f"[:face {format_face(seg.id)}]"
        elif isinstance(seg, Reply):
            content += f"[:refer {seg.id}]"
        elif isinstance(seg, Image):
            content += f"[:image {seg.file}]"
        else:
            content += "[:unsupported]"
    return content


def msglfmt(events: List[GroupMessageEvent], with_id: bool | str = False) -> str:
    current_date = None
    current_time = None
    last_sender = None
    lines: list[str] = []
    for event in events:
        info = ""
        dt = datetime.fromtimestamp(event.time, tz=timezone(TZ))
        ed = dt.date().isoformat()
        et = dt.timetz().isoformat(timespec="minutes")[:5]
        if ed != current_date:
            info += f"[on {ed} {et}]"
        elif et != current_time:
            info += f"[on {et}]"
        current_date, current_time = ed, et
        if event.sender.user_id != last_sender:
            if event.sender.user_id == USR:
                info += f"[from ME {event.sender.user_id} ({event.sender.card or event.sender.nickname})]"
            else:
                info += (
                    f"[from {event.sender.user_id} ({event.sender.card or event.sender.nickname})]"
                )
        last_sender = event.sender.user_id
        if info:
            lines.append(info)
        msg_line = ""
        if with_id is True:
            msg_line += f"[id {event.message_id}]"
        elif with_id == event.message_id:
            msg_line += "[this]"
        msg_line += format_msg(event.message)
        lines.append(msg_line)
    return "\n".join(lines)


__all__ = ["format_msg", "format_face", "parse_face", "parse_msg", "msglfmt"]
