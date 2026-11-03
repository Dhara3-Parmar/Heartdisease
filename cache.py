"""
HeartCare - custom caching for predictions (Streamlit ka built-in cache nahi).

- LRU     : size limit; sabse purani entry hatti hai
- TTL     : entry kuch der baad expire (model/data purana na ho)
- Thread-safe : Lock se
- Single-flight : same input par 10 users ek saath aaye to model sirf 1 baar chalta hai,
                  baaki 9 wahi result ka wait karte hain (concurrent requests efficient)
"""
import time
import threading
from collections import OrderedDict


class _Call:
    """Jo computation chal rahi hai, uska result yahan rakhte hain."""
    def __init__(self):
        self.done = threading.Event()
        self.value = None
        self.error = None


class PredictionCache:

    def __init__(self, max_size=512, ttl=600):
        self.max_size = max_size
        self.ttl = ttl                      
        self._data = OrderedDict()         
        self._inflight = {}               
        self._lock = threading.Lock()
        self.hits = self.misses = self.waits = self.evictions = self.expired = 0

    def get_or_compute(self, key, compute):
        """Cache me ho to wahi do, warna compute() chalao (same key par ek hi baar)."""
        with self._lock:
            item = self._data.get(key)
            if item is not None:
                value, created = item
                if self.ttl is None or time.monotonic() - created < self.ttl:
                    self._data.move_to_end(key)        
                    self.hits += 1
                    return value
                del self._data[key]                      
                self.expired += 1

            call = self._inflight.get(key)
            owner = call is None
            if owner:
                call = self._inflight[key] = _Call()
                self.misses += 1
            else:
                self.waits += 1

        if owner:
            try:
                call.value = compute()                   
            except BaseException as error:
                call.error = error                     
            with self._lock:
                if call.error is None:
                    self._data[key] = (call.value, time.monotonic())
                    self._data.move_to_end(key)
                    while len(self._data) > self.max_size:
                        self._data.popitem(last=False)
                        self.evictions += 1
                del self._inflight[key]
            call.done.set()
        else:
            call.done.wait()

        if call.error is not None:
            raise call.error
        return call.value

    def clear(self):
        with self._lock:
            self._data.clear()

    def stats(self):
        with self._lock:
            total = self.hits + self.misses + self.waits
            return {"size": len(self._data), "hits": self.hits, "misses": self.misses,
                    "waits": self.waits, "evictions": self.evictions, "expired": self.expired,
                    "hit_rate_%": round(100 * (self.hits + self.waits) / total, 1) if total else 0.0}
