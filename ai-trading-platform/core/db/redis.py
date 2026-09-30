import json
from typing import Optional, Any, Dict

import redis.asyncio as redis

from core.config.settings import settings
from core.logging.logger import logger


class RedisManager:
    """
    Central Redis connection manager.

    Two Redis clients are maintained:

    1. `redis`
       - Text/JSON Redis connection.
       - decode_responses=True.
       - Used by market, macro, dashboard, state, kill-switch, etc.

    2. `binary_redis`
       - Binary-safe Redis connection.
       - decode_responses=False.
       - Used for MessagePack/binary experience messages.

    IMPORTANT:
    Never use the text client to publish/read binary MessagePack payloads.
    """

    _instance: Optional["RedisManager"] = None

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.binary_redis: Optional[redis.Redis] = None

    @classmethod
    def get_instance(cls) -> "RedisManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def connect(self):
        """
        Connect both text and binary Redis clients.

        The normal application Redis client remains UTF-8 decoded so existing
        JSON/state consumers continue to behave as before.

        The binary client explicitly disables response decoding so MessagePack
        payloads can safely travel through Redis without UTF-8 conversion.
        """

        # ---------------------------------------------------------
        # TEXT / JSON CLIENT
        # ---------------------------------------------------------
        if self.redis is None:
            try:
                self.redis = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    protocol=2,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )

                await self.redis.ping()

                logger.info(
                    "Successfully connected to Redis "
                    "(text/JSON client)"
                )

            except Exception:
                logger.error(
                    "Failed to connect to Redis text client",
                    exc_info=True,
                )

                self.redis = None

        # ---------------------------------------------------------
        # BINARY CLIENT
        # ---------------------------------------------------------
        if self.binary_redis is None:
            try:
                self.binary_redis = redis.from_url(
                    settings.redis_url,
                    decode_responses=False,
                    protocol=2,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )

                await self.binary_redis.ping()

                logger.info(
                    "Successfully connected to Redis "
                    "(binary-safe client)"
                )

            except Exception:
                logger.error(
                    "Failed to connect to Redis binary client",
                    exc_info=True,
                )

                self.binary_redis = None

    async def disconnect(self):
        """
        Close both Redis connections.
        """

        if self.redis is not None:
            try:
                await self.redis.aclose()
            except Exception:
                logger.error(
                    "Error closing Redis text client",
                    exc_info=True,
                )
            finally:
                self.redis = None

        if self.binary_redis is not None:
            try:
                await self.binary_redis.aclose()
            except Exception:
                logger.error(
                    "Error closing Redis binary client",
                    exc_info=True,
                )
            finally:
                self.binary_redis = None

        logger.info("Disconnected from Redis")

    async def set_state(
        self,
        key: str,
        data: Dict[str, Any],
        ttl_seconds: int = 60,
    ):
        """
        Store normal application state as JSON.

        This deliberately uses the text Redis client.
        """

        if self.redis is None:
            await self.connect()

        if self.redis is None:
            logger.error(
                "Cannot set Redis state: Redis is unavailable"
            )
            return

        try:
            payload = json.dumps(data)

            await self.redis.set(
                key,
                payload,
                ex=ttl_seconds,
            )

            await self.redis.publish(
                key,
                payload,
            )

        except Exception as e:
            logger.error(
                "Failed to set state in Redis",
                extra={
                    "key": key,
                    "error": str(e),
                },
                exc_info=True,
            )

    async def publish_binary(
        self,
        channel: str,
        payload: bytes,
    ) -> int:
        """
        Publish a binary payload safely.

        This method MUST be used for MessagePack/binary messages.

        Redis-py receives the bytes unchanged because this connection uses
        decode_responses=False.
        """

        if not isinstance(payload, bytes):
            raise TypeError(
                "publish_binary() requires a bytes payload"
            )

        if self.binary_redis is None:
            await self.connect()

        if self.binary_redis is None:
            raise RuntimeError(
                "Redis binary client is unavailable"
            )

        try:
            return await self.binary_redis.publish(
                channel,
                payload,
            )

        except Exception as e:
            logger.error(
                "Failed to publish binary Redis message",
                extra={
                    "channel": channel,
                    "payload_size": len(payload),
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    def get_binary_pubsub(self):
        """
        Return a PubSub object backed by the binary-safe Redis client.

        The caller must call `connect()` before using this method.
        """

        if self.binary_redis is None:
            raise RuntimeError(
                "Redis binary client is not connected"
            )

        return self.binary_redis.pubsub()


redis_manager = RedisManager.get_instance()