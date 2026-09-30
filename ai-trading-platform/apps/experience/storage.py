import asyncio

import msgpack

from core.db.redis import redis_manager
from core.logging.logger import logger
from core.ai.replay_buffer import ReplayBuffer


class ExperienceStorageService:
    """
    Experience Storage Service.

    Receives completed trading experiences from Redis and writes them
    into the MySQL-backed ReplayBuffer.

    IMPORTANT:
    Experience messages are encoded using MessagePack and therefore must
    be consumed through Redis with decode_responses=False.
    """

    CHANNEL = "experience:completed"

    def __init__(self):
        # The storage service only writes experiences to MySQL.
        #
        # Keeping the RAM cache disabled prevents the large 200/300-step
        # state sequences from accumulating in memory.
        self.replay_buffer = ReplayBuffer(
            capacity_cache=0
        )

        self.running = False

    async def start(self):
        """
        Start the experience consumer.
        """

        await redis_manager.connect()

        if redis_manager.binary_redis is None:
            logger.error(
                "Could not connect to Redis binary client "
                "for Experience Storage."
            )
            return

        pubsub = redis_manager.get_binary_pubsub()

        await pubsub.subscribe(self.CHANNEL)

        self.running = True

        logger.info(
            "Experience Storage Service listening on "
            f"'{self.CHANNEL}' using binary-safe Redis connection"
        )

        try:
            async for message in pubsub.listen():

                if not self.running:
                    break

                if message.get("type") != "message":
                    continue

                try:
                    raw_data = message.get("data")

                    # -------------------------------------------------
                    # HARD VALIDATION
                    # -------------------------------------------------
                    if not isinstance(raw_data, (bytes, bytearray)):
                        logger.error(
                            "Received non-binary experience payload. "
                            f"Type={type(raw_data).__name__}"
                        )
                        continue

                    if len(raw_data) == 0:
                        logger.warning(
                            "Received empty experience payload."
                        )
                        continue

                    # -------------------------------------------------
                    # MESSAGEPACK DECODE
                    # -------------------------------------------------
                    exp_data = msgpack.unpackb(
                        raw_data,
                        raw=False,
                        strict_map_key=False,
                    )

                    if not isinstance(exp_data, dict):
                        logger.error(
                            "Invalid experience payload: expected dict, "
                            f"received {type(exp_data).__name__}"
                        )
                        continue

                    # -------------------------------------------------
                    # REQUIRED FIELD VALIDATION
                    # -------------------------------------------------
                    required_fields = (
                        "experience_id",
                        "timestamp",
                        "symbol",
                        "action",
                        "reward",
                    )

                    missing_fields = [
                        field
                        for field in required_fields
                        if field not in exp_data
                    ]

                    if missing_fields:
                        logger.error(
                            "Invalid experience payload. "
                            f"Missing fields: {missing_fields}"
                        )
                        continue

                    # -------------------------------------------------
                    # STORE IN MYSQL
                    # -------------------------------------------------
                    self.replay_buffer.add_experience(
                        exp_data
                    )

                    logger.info(
                        "Stored experience | "
                        f"symbol={exp_data.get('symbol')} | "
                        f"action={exp_data.get('action')} | "
                        f"reward={exp_data.get('reward')} | "
                        f"experience_id={exp_data.get('experience_id')}"
                    )

                except msgpack.exceptions.ExtraData as e:
                    logger.error(
                        "Experience MessagePack payload contains "
                        f"extra data: {e}"
                    )

                except (ValueError, TypeError) as e:
                    logger.error(
                        "Invalid experience payload: "
                        f"{e}"
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process experience message: "
                        f"{e}",
                        exc_info=True,
                    )

        except asyncio.CancelledError:
            logger.info(
                "Experience Storage Service cancellation received."
            )
            raise

        except Exception as e:
            logger.error(
                "Experience Storage Service listener crashed: "
                f"{e}",
                exc_info=True,
            )

        finally:
            self.running = False

            try:
                await pubsub.unsubscribe(self.CHANNEL)
            except Exception:
                pass

            try:
                await pubsub.close()
            except Exception:
                pass

            logger.info(
                "Experience Storage Service stopped."
            )

    async def stop(self):
        """
        Stop the experience consumer cleanly.
        """

        self.running = False

        logger.info(
            "Stopping Experience Storage Service..."
        )


async def main():
    service = ExperienceStorageService()

    try:
        await service.start()

    except KeyboardInterrupt:
        logger.info(
            "Keyboard interrupt received."
        )

    finally:
        await service.stop()

        try:
            await redis_manager.disconnect()
        except Exception:
            logger.error(
                "Failed to disconnect Redis cleanly",
                exc_info=True,
            )


if __name__ == "__main__":
    asyncio.run(main())