import asyncio
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI

from agent.news_monitor import DisasterNewsMonitor, create_news_router, normalize_article


def article(title='某地发生地震，救援力量已赶赴现场', url='https://example.gov.cn/news/1'):
    return {'title': title, 'url': url, 'content': '地震灾情最新通报',
            'date': datetime.now(timezone.utc).isoformat(), 'website': '官方通报'}


class ArticleTests(unittest.TestCase):
    def test_freshness_and_disaster_filter(self):
        now = datetime.now(timezone.utc)
        self.assertIsNotNone(normalize_article(article(), now))
        for changes in ({'date': (now-timedelta(days=3)).isoformat()}, {'date': ''},
                        {'url': 'javascript:alert(1)'}, {'url': 'https://user:password@example.com'},
                        {'title': '地震应急演练举行'}, {'date': (now+timedelta(hours=3)).isoformat()}):
            with self.subTest(changes=changes):
                self.assertIsNone(normalize_article({**article(), **changes}, now))
        self.assertIsNone(normalize_article({**article(), 'title': '公司发布财报', 'content': '年度利润增长'}, now))

    def test_naive_china_date_and_tracking_url(self):
        now = datetime.fromisoformat('2026-10-17T04:00:00+00:00')
        item = normalize_article({**article(), 'date': '2026-10-17 11:00:00',
                                  'url': 'https://example.gov.cn/n?id=1&utm_source=app#section'}, now)
        self.assertEqual(item['url'], 'https://example.gov.cn/n?id=1')
        self.assertEqual(item['published_at'], '2026-10-17T11:00:00+08:00')


class MonitorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fetcher = AsyncMock(return_value=[article()])
        self.monitor = DisasterNewsMonitor(self.fetcher, interval=.05)
        self.addAsyncCleanup(self.monitor.close)

    async def wait_until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(.005)
        await asyncio.wait_for(wait(), 2)

    async def test_opt_in_shared_loop_dedup_and_stop(self):
        await asyncio.sleep(.02)
        self.fetcher.assert_not_awaited()
        await self.monitor.subscribe('client-one', True)
        await self.wait_until(lambda: self.monitor.sequence == 1)
        task = self.monitor.task
        await self.monitor.subscribe('client-two', True)
        self.assertIs(task, self.monitor.task)
        await self.wait_until(lambda: self.fetcher.await_count >= 2)
        self.assertEqual(self.monitor.sequence, 1)
        state = self.monitor.snapshot('client-one')
        self.assertEqual(len(state['news']), 1)
        state = self.monitor.snapshot('client-one', state['cursor'], state['epoch'])
        self.assertEqual(state['news'], [])
        await self.monitor.subscribe('client-one', False)
        self.assertIsNotNone(self.monitor.task)
        await self.monitor.subscribe('client-two', False)
        count = self.fetcher.await_count
        await asyncio.sleep(.08)
        self.assertEqual(count, self.fetcher.await_count)
        self.assertIsNone(self.monitor.task)

    async def test_url_and_title_dedup_and_new_event(self):
        self.monitor.clients['client-one'] = self.monitor.clock() + 100
        await self.monitor.check_once()
        self.fetcher.return_value = [article(url='https://example.gov.cn/news/1?utm_source=test'),
                                    article(url='https://other.gov.cn/repost'),
                                    article(title='另一地出现洪涝灾情', url='https://example.gov.cn/news/2')]
        await self.monitor.check_once()
        self.assertEqual(self.monitor.sequence, 2)
        self.assertEqual(len(self.monitor.snapshot('client-one', 1, self.monitor.epoch)['news']), 1)
        self.assertEqual(len(self.monitor.snapshot('client-one', 999, 'previous-process')['news']), 2)

    async def test_failure_retries_without_fake_news(self):
        self.fetcher.side_effect = [RuntimeError('secret provider details'), [article()]]
        await self.monitor.subscribe('client-one', True)
        await self.wait_until(lambda: bool(self.monitor.last_error))
        self.assertEqual(self.monitor.sequence, 0)
        self.assertNotIn('secret', self.monitor.last_error)
        await self.wait_until(lambda: self.monitor.sequence == 1)
        self.assertEqual(self.monitor.last_error, '')

    async def test_disable_cancels_inflight_search(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        async def hanging(now):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        self.monitor.fetcher = hanging
        await self.monitor.subscribe('client-one', True)
        await asyncio.wait_for(started.wait(), 1)
        await self.monitor.subscribe('client-one', False)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.monitor.sequence, 0)

    async def test_abandoned_subscriber_expires(self):
        clock = [0.]
        self.monitor.clock = lambda: clock[0]
        self.monitor.lease_seconds = .01
        await self.monitor.subscribe('client-one', True)
        await self.wait_until(lambda: self.monitor.sequence == 1)
        clock[0] = 1.
        await self.wait_until(lambda: self.monitor.task is None)
        self.assertEqual(self.monitor.clients, {})
        self.assertFalse(self.monitor.snapshot('client-one')['subscribed'])

    async def test_http_subscription_and_updates(self):
        app = FastAPI()
        app.include_router(create_news_router(self.monitor))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            state = await client.get('/api/disaster-news/updates', params={'client_id': 'client-one'})
            self.assertFalse(state.json()['subscribed'])
            self.assertEqual((await client.post('/api/disaster-news/subscriptions', json={'client_id': '../bad', 'enabled': True})).status_code, 422)
            result = await client.post('/api/disaster-news/subscriptions', json={'client_id': 'client-one', 'enabled': True})
            self.assertEqual(result.status_code, 200)
            await self.wait_until(lambda: self.monitor.sequence == 1)
            result = await client.get('/api/disaster-news/updates', params={'client_id': 'client-one', 'after_seq': 0})
            self.assertEqual(result.json()['news'][0]['title'], article()['title'])
            await client.post('/api/disaster-news/subscriptions', json={'client_id': 'client-one', 'enabled': False})
            self.assertIsNone(self.monitor.task)


if __name__ == '__main__':
    unittest.main()
