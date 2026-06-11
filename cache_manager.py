import os
import json
import time
import hashlib
from typing import Optional, Dict, Any

class CacheManager:
    """
    Simple, robust file-based cache.
    Stores JSON responses on disk with automatic expiration.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_filepath(self, key: str) -> str:
        """Generate a safe, flat filename from a cache key using MD5 hash."""
        # Hash prevents any filesystem issues with special characters in keys
        safe_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_hash}.json")
    
    def get(self, key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data if it exists and hasn't expired.
        
        Returns:
            The cached dictionary if valid, otherwise None.
        """
        filepath = self._get_filepath(key)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r') as f:
                cached_data = json.load(f)
            
            # Check if expired
            cached_at = cached_data.get("cached_at", 0)
            if (time.time() - cached_at) > ttl_seconds:
                # Cache is stale, delete it
                os.remove(filepath)
                return None
            
            return cached_data.get("data")
            
        except (json.JSONDecodeError, KeyError, IOError):
            # Corrupted cache file, delete and ignore
            if os.path.exists(filepath):
                os.remove(filepath)
            return None
    
    def set(self, key: str, data: Dict[str, Any]):
        """
        Save data to cache.
        """
        filepath = self._get_filepath(key)
        
        cache_payload = {
            "cached_at": time.time(),
            "data": data
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(cache_payload, f)
        except IOError as e:
            print(f"Cache write error for {key}: {e}")
    
    def clear(self):
        """Delete all cached files."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, filename))