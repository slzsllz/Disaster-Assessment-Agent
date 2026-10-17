"""Shared, opt-in disaster news polling for connected frontend subscribers."""
from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import re
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, StrictBool

LOGGER = logging.getLogger(__name__)
CHINA = ZoneInfo('Asia/Shanghai')
CLIENT_PATTERN = r'^[a-zA-Z0-9_-]{8,100}$'
DISASTER_WORDS = re.compile(r'地震|洪水|洪涝|暴雨|台风|山洪|滑坡|泥石流|森林火灾|山火|灾情|受灾|震级|震源|人员伤亡|地质灾害')
NON_NEWS = re.compile(r'演练|科普|知识问答|防范指南|安全常识|历史回顾')


def normalize_article(reference: dict, now: datetime) -> dict | None:
    """Only emit dated, recent disaster reports with safe source links."""
    title = str(reference.get('title') or '').strip()
    content = str(reference.get('content') or '')
    if not title or not DISASTER_WORDS.search(title + content) or NON_NEWS.search(title):
        return None
    try:
        link = urlsplit(str(reference.get('url') or ''))
        if link.scheme not in ('http', 'https') or not link.hostname or link.username or link.password:
            return None
        query = [(k, v) for k, v in parse_qsl(link.query, keep_blank_values=True)
                 if not k.lower().startswith('utm_') and k.lower() not in ('spm', 'fbclid', 'gclid')]
        url = urlunsplit((link.scheme.lower(), link.netloc.lower(), link.path, urlencode(query), ''))
        date = str(reference.get('date') or '').strip()
        published = datetime.fromisoformat(date.replace('Z', '+00:00'))
        if published.tzinfo is None:
            published = published.replace(tzinfo=CHINA)
        if not now - timedelta(hours=24) <= published <= now + timedelta(minutes=10):
            return None
    except (ValueError, TypeError):
        return None
    normalized_title = re.sub(r'[\W_]+', '', title).casefold()
    return {
        'id': hashlib.sha256(url.encode()).hexdigest()[:24],
        'title': title[:300], 'url': url,
        'summary': re.sub(r'[\x00-\x08\x0b-\x1f]', '', content)[:600],
        'source': str(reference.get('website') or link.hostname)[:150],
        'published_at': published.isoformat(),
        '_title_key': 'title:' + normalized_title,
    }


async def fetch_disaster_news(now: datetime) -> list[dict]:
    # Reuse the configured search transport without creating per-minute artifacts
    # or invoking an LLM/agent for each poll.
    from agent.tools.WebSearch import _request_search

    today = now.astimezone(CHINA).strftime('%Y年%m月%d日')
    data = await _request_search({
        'messages': [{'role': 'user', 'content': f'{today} 最新灾情新闻 地震 洪涝 台风 山洪 滑坡 森林火灾 官方通报'}],
        'search_source': 'baidu_search_v2',
        'resource_type_filter': [{'type': 'web', 'top_k': 20}],
        'search_recency_filter': 'week',
    })
    return data['references']


class DisasterNewsMonitor:
    def __init__(self, fetcher=fetch_disaster_news, interval=60., lease_seconds=180., *, clock=time.monotonic):
        self.fetcher = fetcher
        self.interval = interval
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.clients: dict[str, float] = {}
        self.events = deque(maxlen=200)
        self.seen = OrderedDict()
        self.sequence = 0
        self.epoch = uuid4().hex
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.last_checked = None
        self.last_success = None
        self.last_error = ''
        self.checking = False

    def _prune(self):
        now = self.clock()
        for client in list(self.clients):
            if self.clients[client] <= now:
                self.clients.pop(client, None)

    async def subscribe(self, client_id: str, enabled: bool):
        async with self.lock:
            self._prune()
            if enabled:
                self.clients[client_id] = self.clock() + self.lease_seconds
                if self.task is None or self.task.done():
                    self.task = asyncio.create_task(self._run(), name='disaster-news-monitor')
            else:
                self.clients.pop(client_id, None)
                if not self.clients:
                    await self._stop()
            return self.snapshot(client_id)

    def snapshot(self, client_id: str, after_seq=0, epoch=None):
        self._prune()
        subscribed = client_id in self.clients
        if subscribed:
            self.clients[client_id] = self.clock() + self.lease_seconds
        if epoch != self.epoch or after_seq > self.sequence:
            after_seq = 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        items = [item for item in self.events if item['sequence'] > after_seq
                 and datetime.fromisoformat(item['published_at']) >= cutoff] if subscribed else []
        return {
            'subscribed': subscribed, 'interval_seconds': self.interval,
            'epoch': self.epoch, 'cursor': self.sequence, 'news': items,
            'checking': self.checking, 'last_checked_at': self.last_checked,
            'last_success_at': self.last_success, 'error': self.last_error,
        }

    async def check_once(self):
        self.checking = True
        now = datetime.now(timezone.utc)
        self.last_checked = now.isoformat()
        try:
            references = await self.fetcher(now)
            # A disabled/expired subscription must not generate late notifications.
            self._prune()
            if not self.clients:
                return
            articles = [normalize_article(ref, now) for ref in references if isinstance(ref, dict)]
            articles = sorted((a for a in articles if a), key=lambda a: datetime.fromisoformat(a['published_at']).timestamp())
            for article in articles:
                title_key = article.pop('_title_key')
                url_key = 'url:' + article['id']
                if title_key in self.seen or url_key in self.seen:
                    continue
                self.seen[title_key] = self.seen[url_key] = None
                while len(self.seen) > 4000:
                    self.seen.popitem(last=False)
                self.sequence += 1
                self.events.append({**article, 'sequence': self.sequence, 'detected_at': now.isoformat()})
            self.last_success = now.isoformat()
            self.last_error = ''
        except asyncio.CancelledError:
            raise
        except Exception:
            # The frontend gets an actionable status, never provider credentials or tracebacks.
            self.last_error = '灾情检索暂时失败，将在下一分钟重试；请检查搜索服务配置或网络。'
            LOGGER.warning('Disaster news search failed; retrying at the next interval.')
        finally:
            self.checking = False

    async def _run(self):
        try:
            while True:
                self._prune()
                if not self.clients:
                    return
                started = self.clock()
                await self.check_once()
                await asyncio.sleep(max(0., self.interval - (self.clock() - started)))
        finally:
            if self.task is asyncio.current_task():
                self.task = None

    async def _stop(self):
        task, self.task = self.task, None
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def close(self):
        async with self.lock:
            self.clients.clear()
            await self._stop()


class NewsSubscription(BaseModel):
    client_id: str = Field(min_length=8, max_length=100, pattern=CLIENT_PATTERN)
    enabled: StrictBool


def create_news_router(monitor: DisasterNewsMonitor) -> APIRouter:
    router = APIRouter(prefix='/api/disaster-news', tags=['disaster-news'])

    @router.post('/subscriptions')
    async def subscribe(subscription: NewsSubscription):
        return await monitor.subscribe(subscription.client_id, subscription.enabled)

    @router.get('/updates')
    async def updates(client_id: str = Query(pattern=CLIENT_PATTERN),
                      after_seq: int = Query(default=0, ge=0), epoch: str | None = None):
        return monitor.snapshot(client_id, after_seq, epoch)

    return router
