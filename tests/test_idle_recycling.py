import asyncio
import unittest
from unittest.mock import patch

import app.main as main


class ActiveRequestsTest(unittest.IsolatedAsyncioTestCase):
    async def test_inflight_websocket_remains_active_until_handler_finishes(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def handler(scope, receive, send):
            entered.set()
            await release.wait()

        middleware = main.ActiveRequestsMiddleware(handler)
        scope = {"type": "websocket", "path": "/v1/transcriptions/stream"}
        with patch.object(main.engine, "touch") as touch:
            task = asyncio.create_task(middleware(scope, None, None))
            await asyncio.wait_for(entered.wait(), timeout=1)
            self.assertEqual(main.active_requests, 1)
            release.set()
            await asyncio.wait_for(task, timeout=1)
            self.assertEqual(main.active_requests, 0)
            touch.assert_called_once()

    async def test_failed_upload_releases_request_and_refreshes_activity(self):
        async def handler(scope, receive, send):
            raise ValueError("failed")

        middleware = main.ActiveRequestsMiddleware(handler)
        scope = {"type": "http", "path": "/v1/transcriptions"}
        with patch.object(main.engine, "touch") as touch:
            with self.assertRaises(ValueError):
                await middleware(scope, None, None)
            self.assertEqual(main.active_requests, 0)
            touch.assert_called_once()

    async def test_complete_upload_yields_while_request_is_protected(self):
        pending = asyncio.Event()
        proceed = asyncio.Event()

        async def receive():
            asyncio.get_running_loop().call_soon(pending.set)
            return {"type": "http.request", "body": b"audio", "more_body": False}

        async def handler(scope, wrapped_receive, send):
            await wrapped_receive()
            await proceed.wait()

        middleware = main.ActiveRequestsMiddleware(handler)
        scope = {"type": "http", "path": "/v1/transcriptions", "method": "POST"}
        task = asyncio.create_task(middleware(scope, receive, None))
        await asyncio.wait_for(pending.wait(), timeout=1)
        self.assertEqual(main.active_requests, 1)
        proceed.set()
        await asyncio.wait_for(task, timeout=1)
        self.assertEqual(main.active_requests, 0)


if __name__ == "__main__":
    unittest.main()
