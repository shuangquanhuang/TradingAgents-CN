"""
Xueqiu data fetcher through a local Chrome MCP server.

The server is expected to expose a streamable HTTP MCP endpoint, for example:
http://127.0.0.1:12306/mcp
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_shared_client: Optional["XueqiuMCPClient"] = None
_shared_client_lock = threading.RLock()
_session_file_lock = threading.RLock()


@dataclass
class XueqiuItem:
    """Normalized Xueqiu stock status/news item."""

    symbol: str
    title: str
    content: str
    summary: str
    url: str
    source: str
    author: str
    publish_time: datetime
    sentiment: str
    sentiment_score: float
    importance: str
    category: str
    data_source: str
    metrics: Dict[str, Any]
    raw_type: str = ""

    def to_news_dict(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        code = symbol or self.symbol
        return {
            "symbol": code,
            "symbols": [code],
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "url": self.url,
            "source": self.source,
            "author": self.author,
            "publish_time": self.publish_time,
            "category": self.category,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "importance": self.importance,
            "keywords": _extract_keywords(f"{self.title} {self.content}"),
            "data_source": self.data_source,
            "metrics": self.metrics,
        }


def normalize_xueqiu_symbol(symbol: str) -> str:
    """Convert common A-share/HK formats to Xueqiu symbols."""
    raw = str(symbol or "").strip().upper()
    if raw.startswith(("SH", "SZ", "HK")):
        return raw
    if raw.endswith(".SH") or raw.endswith(".SS") or raw.endswith(".XSHG"):
        return f"SH{raw.split('.')[0]}"
    if raw.endswith(".SZ") or raw.endswith(".XSHE"):
        return f"SZ{raw.split('.')[0]}"
    if raw.endswith(".HK"):
        return f"HK{raw.split('.')[0].zfill(5)}"
    if re.fullmatch(r"\d{6}", raw):
        return f"SH{raw}" if raw.startswith(("60", "68", "90")) else f"SZ{raw}"
    if re.fullmatch(r"\d{4,5}", raw):
        return f"HK{raw.zfill(5)}"
    return raw


def normalize_storage_symbol(symbol: str) -> str:
    """Normalize symbol for local MongoDB storage/query."""
    xq = normalize_xueqiu_symbol(symbol)
    if re.fullmatch(r"(SH|SZ)\d{6}", xq):
        return xq[2:]
    if re.fullmatch(r"HK\d{5}", xq):
        return xq[2:]
    return str(symbol or "").strip().upper()


class XueqiuMCPClient:
    """Small MCP client for the local Chrome MCP server."""

    def __init__(self, endpoint: Optional[str] = None, timeout: int = 30):
        self.endpoint = endpoint or os.getenv("XUEQIU_MCP_URL", "http://127.0.0.1:12306/mcp")
        self.timeout = timeout
        self.session = requests.Session()
        self.session_id: Optional[str] = os.getenv("XUEQIU_MCP_SESSION_ID") or _read_session_id()
        self._request_id = int(time.time() * 1000)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _parse_sse(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}

        lines = text.splitlines()
        data_lines: List[str] = []
        collecting = False

        for line in lines:
            if line.startswith("event:"):
                collecting = False
                continue
            if line.startswith("data:"):
                collecting = True
                data_lines.append(line[5:].lstrip())
                continue
            if collecting and line:
                # Some Chrome MCP responses contain large JSON strings and are
                # split across physical lines without repeating "data:".
                data_lines.append(line)

        if data_lines:
            payload = "\n".join(data_lines).strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                compact_payload = "".join(data_lines).strip()
                return json.loads(compact_payload)

        return json.loads(text)

    def _post(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        headers = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id

        response = self.session.post(
            self.endpoint,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=timeout or self.timeout,
        )
        if not self.session_id:
            self.session_id = response.headers.get("mcp-session-id")
            if self.session_id:
                _write_session_id(self.session_id)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"MCP请求失败: HTTP {response.status_code} {response.text[:500]}") from exc
        if not response.text.strip():
            return {}
        return self._parse_sse(response.text)

    def close(self) -> None:
        """Close the streamable HTTP MCP session if the server supports DELETE."""
        if not self.session_id:
            return
        try:
            self.session.delete(
                self.endpoint,
                headers={"mcp-session-id": self.session_id},
                timeout=5,
            )
        except Exception as exc:
            logger.debug("Failed to close Xueqiu MCP session: %s", exc)
        finally:
            _clear_session_id()
            self.session_id = None

    def initialize(self) -> None:
        if self.session_id:
            try:
                self._post({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list"}, timeout=8)
                return
            except Exception as exc:
                logger.warning("Saved Xueqiu MCP session is not reusable, clearing it: %s", exc)
                self.session_id = None
                _clear_session_id()
        self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "tradingagents-xueqiu", "version": "1.0.0"},
                },
            }
        )
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception as exc:
            logger.debug("Xueqiu MCP initialized notification failed: %s", exc)

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Any:
        self.initialize()
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
        result = self._post(payload, timeout=timeout)
        if "error" in result:
            raise RuntimeError(result["error"])
        content = result.get("result", {}).get("content", [])
        if not content:
            return result
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        try:
            return json.loads(text)
        except Exception:
            return text

    def fetch_stock_items(self, symbol: str, limit: int = 20) -> List[XueqiuItem]:
        xq_symbol = normalize_xueqiu_symbol(symbol)
        storage_symbol = normalize_storage_symbol(symbol)
        logger.info("Fetching Xueqiu items through MCP: %s -> %s", symbol, xq_symbol)

        self.call_tool(
            "chrome_navigate",
            {"url": f"https://xueqiu.com/S/{xq_symbol}", "width": 1280, "height": 900},
            timeout=20,
        )
        self.call_tool("chrome_computer", {"action": "wait", "duration": 6}, timeout=15)

        status_url = self._find_status_url_from_performance(xq_symbol)
        capture = {}

        if not status_url:
            try:
                self.call_tool(
                    "chrome_network_capture",
                    {
                        "action": "start",
                        "needResponseBody": True,
                        "includeStatic": False,
                        "maxCaptureTime": 60000,
                        "inactivityTimeout": 8000,
                    },
                    timeout=20,
                )
                self.call_tool("chrome_navigate", {"refresh": True}, timeout=20)
                self.call_tool("chrome_computer", {"action": "wait", "duration": 6}, timeout=15)
                capture = self.call_tool("chrome_network_capture", {"action": "stop"}, timeout=30)
            except Exception as exc:
                logger.warning("Xueqiu MCP network capture failed, using direct status URL fallback: %s", exc)
                capture = {}

            status_url = self._find_status_url(capture, xq_symbol)
        else:
            capture = self._capture_status_response(xq_symbol)

        if not status_url:
            # One more cheap attempt after any reload/capture attempt.
            status_url = self._find_status_url_from_performance(xq_symbol)

        if not status_url:
            raise RuntimeError(f"未捕获到雪球 {xq_symbol} 的动态接口")

        body = self._find_status_body(capture, xq_symbol)
        if body is None:
            try:
                body = self._request_status_json(status_url, xq_symbol)
            except Exception as exc:
                logger.warning("Xueqiu status API blocked, falling back to rendered DOM extraction: %s", exc)
                return self._fetch_rendered_page_items(xq_symbol, storage_symbol, limit)
        raw_items = body.get("list", []) if isinstance(body, dict) else []
        if not raw_items:
            return self._fetch_rendered_page_items(xq_symbol, storage_symbol, limit)
        return [_normalize_status_item(item, storage_symbol) for item in raw_items[: max(1, int(limit))]]

    def _fetch_rendered_page_items(self, xq_symbol: str, storage_symbol: str, limit: int) -> List[XueqiuItem]:
        """Extract visible rendered Xueqiu discussion/news items from the browser DOM."""
        try:
            self.call_tool("chrome_computer", {"action": "scroll", "scrollDirection": "down", "scrollAmount": 6}, timeout=10)
            self.call_tool("chrome_computer", {"action": "wait", "duration": 1}, timeout=8)
        except Exception as exc:
            logger.debug("Failed to scroll Xueqiu page before DOM extraction: %s", exc)

        js_result = self.call_tool(
            "chrome_javascript",
            {
                "code": f"""
const symbol = {json.dumps(xq_symbol)};
const normalized = symbol.replace(/^SH|^SZ|^HK/, '');
const seen = new Set();
const results = [];
const blocks = Array.from(document.querySelectorAll('article, section, div, li'));
for (const el of blocks) {{
  const text = (el.innerText || '').replace(/\\s+/g, ' ').trim();
  if (!text || text.length < 12 || text.length > 2500) continue;
  if (!text.includes(symbol) && !text.includes(normalized) && !text.includes('博睿数据')) continue;
  if (seen.has(text)) continue;
  const linkEl = el.querySelector('a[href*="/"]');
  let href = linkEl ? linkEl.getAttribute('href') : '';
  if (href && href.startsWith('/')) href = location.origin + href;
  const authorEl = el.querySelector('a[href^="/u/"], a[href^="/"][href*=""]');
  const author = authorEl ? (authorEl.innerText || '').trim() : '';
  seen.add(text);
  results.push({{
    title: text.slice(0, 80),
    text,
    url: href || location.href,
    author,
    source: '雪球页面',
    created_at: Date.now(),
    like_count: 0,
    reply_count: 0,
    retweet_count: 0,
    view_count: 0,
    type: text.includes('公告') || text.includes('新闻') || text.includes('减持') ? '3' : '0'
  }});
  if (results.length >= {max(1, int(limit))}) break;
}}
return results;
""",
                "timeoutMs": 20000,
                "maxOutputBytes": 120000,
            },
            timeout=30,
        )
        raw_result = js_result.get("result") if isinstance(js_result, dict) else js_result
        if isinstance(raw_result, str):
            raw_items = json.loads(raw_result)
        else:
            raw_items = raw_result or []
        items = [_normalize_status_item(item, storage_symbol) for item in raw_items[: max(1, int(limit))]]
        logger.info("从雪球渲染页面提取资讯: %s 条", len(items))
        return items

    def _capture_status_response(self, xq_symbol: str) -> Dict[str, Any]:
        try:
            self.call_tool(
                "chrome_network_capture",
                {
                    "action": "start",
                    "needResponseBody": True,
                    "includeStatic": False,
                    "maxCaptureTime": 60000,
                    "inactivityTimeout": 8000,
                },
                timeout=20,
            )
            self.call_tool("chrome_navigate", {"refresh": True}, timeout=20)
            self.call_tool("chrome_computer", {"action": "wait", "duration": 6}, timeout=15)
            return self.call_tool("chrome_network_capture", {"action": "stop"}, timeout=30)
        except Exception as exc:
            logger.warning("Xueqiu MCP response-body capture failed: %s", exc)
            return {}

    def _find_status_body(self, capture: Any, xq_symbol: str) -> Optional[Dict[str, Any]]:
        if not isinstance(capture, dict):
            return None

        def iter_dicts(value: Any):
            if isinstance(value, dict):
                yield value
                for child in value.values():
                    yield from iter_dicts(child)
            elif isinstance(value, list):
                for child in value:
                    yield from iter_dicts(child)

        for node in iter_dicts(capture):
            url = str(node.get("url", ""))
            if "/query/v1/symbol/search/status.json" not in url:
                continue
            if xq_symbol not in url and "symbol=" in url:
                continue
            for key in ("body", "responseBody", "response_body", "content", "text"):
                parsed = self._parse_json_body(node.get(key))
                if parsed and isinstance(parsed.get("list"), list):
                    logger.info("使用 MCP 网络捕获响应体获取雪球资讯: %s 条", len(parsed.get("list", [])))
                    return parsed
        return None

    def _request_status_json(self, status_url: str, xq_symbol: str) -> Dict[str, Any]:
        api_result = self.call_tool(
            "chrome_network_request",
            {
                "url": status_url,
                "method": "GET",
                "headers": {
                    "accept": "application/json, text/plain, */*",
                    "referer": f"https://xueqiu.com/S/{xq_symbol}",
                },
                "timeout": 30000,
            },
            timeout=40,
        )
        response = (api_result or {}).get("response", {})
        body = response.get("body")
        parsed = self._parse_json_body(body)
        if parsed is not None:
            return parsed

        logger.warning(
            "Xueqiu MCP network_request returned non-json body, status=%s content-type=%s preview=%r; trying in-page fetch",
            response.get("status"),
            (response.get("headers") or {}).get("content-type"),
            str(body)[:120],
        )
        return self._request_status_json_in_page(status_url)

    def _request_status_json_in_page(self, status_url: str) -> Dict[str, Any]:
        js_result = self.call_tool(
            "chrome_javascript",
            {
                "code": f"""
const r = await fetch({json.dumps(status_url)}, {{
  headers: {{'accept': 'application/json, text/plain, */*'}},
  credentials: 'include'
}});
const text = await r.text();
return {{status: r.status, contentType: r.headers.get('content-type'), text}};
""",
                "timeoutMs": 30000,
                "maxOutputBytes": 200000,
            },
            timeout=40,
        )
        result = js_result.get("result") if isinstance(js_result, dict) else js_result
        if isinstance(result, str):
            result = json.loads(result)
        text = (result or {}).get("text", "")
        parsed = self._parse_json_body(text)
        if parsed is not None:
            return parsed
        raise RuntimeError(
            f"雪球接口未返回JSON: status={(result or {}).get('status')} "
            f"content_type={(result or {}).get('contentType')} body={text[:200]!r}"
        )

    def _parse_json_body(self, body: Any) -> Optional[Dict[str, Any]]:
        if isinstance(body, dict):
            return body
        if isinstance(body, str):
            text = body.strip()
            if not text:
                return None
            if text.startswith("<"):
                return None
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _find_status_url_from_performance(self, xq_symbol: str) -> Optional[str]:
        try:
            js_result = self.call_tool(
                "chrome_javascript",
                {
                    "code": """
const urls = performance.getEntriesByType('resource')
  .map((entry) => entry.name)
  .filter((url) => url.includes('/query/v1/symbol/search/status.json'));
return urls;
""",
                    "timeoutMs": 10000,
                    "maxOutputBytes": 20000,
                },
                timeout=15,
            )
            raw_result = js_result.get("result") if isinstance(js_result, dict) else js_result
            if isinstance(raw_result, str):
                try:
                    urls = json.loads(raw_result)
                except Exception:
                    urls = re.findall(r"https://[^\"'\s]+/query/v1/symbol/search/status\.json[^\"'\s]+", raw_result)
            else:
                urls = raw_result or []
            for url in urls:
                if xq_symbol in url:
                    return url
            return urls[0] if urls else None
        except Exception as exc:
            logger.debug("Failed to read Xueqiu status URL from performance entries: %s", exc)
            return None

    def _find_status_url(self, capture: Any, xq_symbol: str) -> Optional[str]:
        if isinstance(capture, dict):
            requests_list = capture.get("requests", [])
        else:
            requests_list = []
        for request in requests_list:
            url = request.get("url", "")
            if "/query/v1/symbol/search/status.json" in url and xq_symbol in url:
                return url
        for request in requests_list:
            url = request.get("url", "")
            if "/query/v1/symbol/search/status.json" in url:
                return url
        return None


def fetch_xueqiu_stock_items(symbol: str, limit: int = 20, endpoint: Optional[str] = None) -> List[XueqiuItem]:
    """Fetch normalized Xueqiu items. Raises when the MCP endpoint is unavailable."""
    global _shared_client
    with _shared_client_lock:
        client_endpoint = endpoint or os.getenv("XUEQIU_MCP_URL", "http://127.0.0.1:12306/mcp")
        if _shared_client is None or _shared_client.endpoint != client_endpoint:
            _shared_client = XueqiuMCPClient(endpoint=client_endpoint)

        try:
            return _shared_client.fetch_stock_items(symbol, limit=limit)
        except Exception as exc:
            # If the saved session became invalid, reset once and retry.
            message = str(exc)
            if any(text in message for text in ("Invalid or missing MCP session ID", "Session not found")):
                logger.warning("Xueqiu MCP session invalid, resetting client and retrying once: %s", exc)
                try:
                    _shared_client.close()
                finally:
                    _shared_client = XueqiuMCPClient(endpoint=client_endpoint)
                return _shared_client.fetch_stock_items(symbol, limit=limit)
            raise


def _session_file_path() -> Path:
    default_path = Path(__file__).resolve().parents[2] / "data" / "xueqiu_mcp_session.txt"
    return Path(os.getenv("XUEQIU_MCP_SESSION_FILE", str(default_path)))


def _read_session_id() -> Optional[str]:
    with _session_file_lock:
        try:
            path = _session_file_path()
            if path.exists():
                session_id = path.read_text(encoding="utf-8").strip()
                return session_id or None
        except Exception as exc:
            logger.debug("Failed to read Xueqiu MCP session file: %s", exc)
        return None


def _write_session_id(session_id: str) -> None:
    with _session_file_lock:
        try:
            path = _session_file_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(session_id, encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to write Xueqiu MCP session file: %s", exc)


def _clear_session_id() -> None:
    with _session_file_lock:
        try:
            path = _session_file_path()
            if path.exists():
                path.unlink()
        except Exception as exc:
            logger.debug("Failed to clear Xueqiu MCP session file: %s", exc)


def format_items_for_news(items: List[XueqiuItem], symbol: str) -> str:
    if not items:
        return ""
    lines = [
        f"# {symbol} 雪球资讯",
        "",
        f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"资讯数量: {len(items)} 条",
        "",
    ]
    for index, item in enumerate(items, 1):
        lines.append(f"## {index}. {item.title}")
        lines.append(
            f"来源: {item.source} | 作者: {item.author or '-'} | 时间: {item.publish_time.strftime('%Y-%m-%d %H:%M')}"
        )
        lines.append(f"情绪: {item.sentiment} ({item.sentiment_score:+.2f}) | 互动: {item.metrics}")
        if item.summary:
            lines.append(item.summary)
        lines.append(f"链接: {item.url}")
        lines.append("")
    return "\n".join(lines).strip()


def format_items_for_sentiment(items: List[XueqiuItem], symbol: str) -> str:
    if not items:
        return ""
    positive = sum(1 for item in items if item.sentiment == "positive")
    negative = sum(1 for item in items if item.sentiment == "negative")
    neutral = len(items) - positive - negative
    avg_score = sum(item.sentiment_score for item in items) / len(items)
    total_replies = sum(int(item.metrics.get("replies") or 0) for item in items)
    total_likes = sum(int(item.metrics.get("likes") or 0) for item in items)
    total_views = sum(int(item.metrics.get("views") or 0) for item in items)

    lines = [
        f"# {symbol} 雪球用户情绪",
        "",
        f"获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"样本数量: {len(items)} 条",
        f"情绪分布: 正面 {positive} / 中性 {neutral} / 负面 {negative}",
        f"平均情绪分: {avg_score:+.2f}",
        f"互动热度: 点赞 {total_likes}，回复 {total_replies}，浏览 {total_views}",
        "",
        "## 代表性讨论",
    ]
    for index, item in enumerate(items[:10], 1):
        lines.append(f"{index}. [{item.sentiment} {item.sentiment_score:+.2f}] {item.title}")
        lines.append(f"   来源/作者: {item.source} / {item.author or '-'}，时间: {item.publish_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"   摘要: {item.summary}")
        lines.append(f"   链接: {item.url}")
    return "\n".join(lines).strip()


def _normalize_status_item(item: Dict[str, Any], symbol: str) -> XueqiuItem:
    text = _strip_html(item.get("description") or item.get("text") or "")
    title = _strip_html(item.get("title") or "") or text[:80] or "雪球讨论"
    source = item.get("source") or "雪球"
    user = item.get("user") or {}
    author = user.get("screen_name", "")
    created_at = item.get("created_at")
    publish_time = datetime.utcnow()
    if isinstance(created_at, (int, float)) and created_at > 0:
        publish_time = datetime.fromtimestamp(created_at / 1000)

    score = _score_sentiment(f"{title} {text}")
    sentiment = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
    url = item.get("target") or ""
    if url and url.startswith("/"):
        url = f"https://xueqiu.com{url}"

    return XueqiuItem(
        symbol=symbol,
        title=title,
        content=text,
        summary=text[:220] + "..." if len(text) > 220 else text,
        url=url,
        source=source,
        author=author,
        publish_time=publish_time,
        sentiment=sentiment,
        sentiment_score=score,
        importance=_assess_importance(title, item),
        category=_classify_category(title),
        data_source="xueqiu_mcp",
        metrics={
            "likes": item.get("like_count", 0),
            "replies": item.get("reply_count", 0),
            "retweets": item.get("retweet_count", 0),
            "views": item.get("view_count", 0),
            "favorites": item.get("fav_count", 0),
        },
        raw_type=str(item.get("type", "")),
    )


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return _repair_mojibake(html.unescape(re.sub(r"\s+", " ", text)).strip())


def repair_mojibake_text(value: Any) -> str:
    """Public helper for repairing UTF-8 text decoded as Latin-1."""
    return _repair_mojibake(str(value or ""))


def _repair_mojibake(value: str) -> str:
    if not value:
        return value

    # Common signature of Chinese UTF-8 bytes decoded as latin-1/cp1252:
    # "åç¿æ°æ®" should be "博睿数据".
    suspicious_chars = ("å", "æ", "ç", "è", "ä", "ï¼", "ã", "€")
    if not any(char in value for char in suspicious_chars):
        return value

    candidates = [value]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(value.encode(encoding, errors="ignore").decode("utf-8", errors="ignore"))
        except Exception:
            continue

    def score(text: str) -> int:
        chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        suspicious = sum(text.count(ch) for ch in suspicious_chars)
        return chinese * 3 - suspicious * 2 + len(text)

    return max(candidates, key=score)


def _score_sentiment(text: str) -> float:
    positive_words = ["上涨", "涨停", "利好", "突破", "增长", "盈利", "超预期", "修复", "看好", "机会", "买入"]
    negative_words = ["下跌", "跌停", "利空", "亏损", "减持", "风险", "下滑", "警告", "卖出", "死心", "认输"]
    positive = sum(text.count(word) for word in positive_words)
    negative = sum(text.count(word) for word in negative_words)
    total = positive + negative
    if total == 0:
        return 0.0
    return max(-1.0, min(1.0, (positive - negative) / max(total, 3)))


def _classify_category(title: str) -> str:
    if any(word in title for word in ["公告", "减持", "增持", "业绩", "财报", "年报", "季报"]):
        return "company_announcement"
    if any(word in title for word in ["行业", "软件", "AI", "人工智能", "市场"]):
        return "industry_news"
    return "social_discussion"


def _assess_importance(title: str, item: Dict[str, Any]) -> str:
    if any(word in title for word in ["重大", "减持", "业绩", "亏损", "涨停", "公告"]):
        return "high"
    engagement = sum(int(item.get(key) or 0) for key in ("like_count", "reply_count", "retweet_count"))
    if engagement >= 10:
        return "medium"
    return "low"


def _extract_keywords(text: str) -> List[str]:
    keywords = [
        "减持",
        "增持",
        "业绩",
        "亏损",
        "盈利",
        "AI",
        "软件",
        "可观测性",
        "财报",
        "公告",
        "风险",
        "机会",
    ]
    return [word for word in keywords if word in text][:10]
