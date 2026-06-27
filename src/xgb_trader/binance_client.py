from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests


@dataclass
class BinanceFuturesClient:
    base_url: str
    api_key: str | None = None
    api_secret: str | None = None
    min_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        self._last_request_at = 0.0
        self._server_time_offset_ms: int | None = None
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _wait(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

    def _server_timestamp_ms(self) -> int:
        if self._server_time_offset_ms is None:
            self._wait()
            response = self.session.get(f"{self.base_url}/fapi/v1/time", timeout=30)
            self._last_request_at = time.time()
            response.raise_for_status()
            server_time = int(response.json()["serverTime"])
            self._server_time_offset_ms = server_time - int(time.time() * 1000)
        return int(time.time() * 1000) + self._server_time_offset_ms

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        params = dict(params or {})
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("Signed Binance request requires BINANCE_API_KEY and BINANCE_API_SECRET.")
            params.setdefault("recvWindow", 5000)
            params["timestamp"] = self._server_timestamp_ms()
            query = urlencode(params, doseq=True)
            params["signature"] = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

        self._wait()
        response = self.session.request(method, f"{self.base_url}{path}", params=params, timeout=30)
        self._last_request_at = time.time()

        if response.status_code in {418, 429}:
            retry_after = int(response.headers.get("Retry-After", "60"))
            time.sleep(retry_after)
            return self._request(method, path, params, signed=False)

        response.raise_for_status()
        return response.json()

    def get(self, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        return self._request("GET", path, params=params, signed=signed)

    def post(self, path: str, params: dict[str, Any] | None = None, signed: bool = True) -> Any:
        return self._request("POST", path, params=params, signed=signed)

    def delete(self, path: str, params: dict[str, Any] | None = None, signed: bool = True) -> Any:
        return self._request("DELETE", path, params=params, signed=signed)
