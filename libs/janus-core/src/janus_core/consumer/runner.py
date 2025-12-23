"""
Consumer Entry Point - Janus Core

Run the Cortex consumer (memory consolidation worker) with graceful shutdown.

Usage:
    janus-consumer  # Via entry point
    python -m janus_core.consumer.runner
"""

import asyncio
import logging
import signal
from typing import Optional

import redis.asyncio as redis

from janus_core.config import settings
from janus_core.db.connection import init_db_pool, close_db_pool
from janus_core.db.stm import STMManager

logger = logging.getLogger(__name__)


class CortexConsumer:
    """
    Async consumer for memory consolidation.

    Reads from Redis Streams (STM) and triggers consolidation
    when enough turns have accumulated or topic boundaries are detected.
    """

    def __init__(self):
        self.running = False
        self.redis: Optional[redis.Redis] = None
        self.stm: Optional[STMManager] = None

    async def initialize(self) -> None:
        """Initialize all dependencies."""
        logger.info("Initializing Cortex Consumer...")

        # Initialize database pool
        await init_db_pool()

        # Initialize Redis
        self.redis = redis.from_url(settings.REDIS_URL, decode_responses=True)

        # Initialize STM manager
        self.stm = STMManager(self.redis)

        logger.info("Cortex Consumer initialized")

    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down Cortex Consumer...")

        if self.redis:
            await self.redis.close()

        await close_db_pool()

        logger.info("Cortex Consumer shutdown complete")

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self.running = False

    async def run(self) -> None:
        """Main consumer loop.

        WARNING: This is a stub implementation. The actual XREADGROUP-based
        consumer loop is not yet implemented. This currently just sleeps
        to allow graceful shutdown testing.
        """
        self.running = True

        try:
            await self.initialize()

            logger.warning(
                "Cortex Consumer started but consolidation loop is NOT IMPLEMENTED. "
                "This consumer will not process any sessions."
            )

            while self.running:
                # TODO: Implement actual XREADGROUP-based consumer loop
                # See: https://redis.io/commands/xreadgroup/
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Consumer task cancelled")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
        finally:
            await self.shutdown()


def main():
    """
    Main entry point for the Cortex consumer.

    Sets up signal handlers for graceful shutdown and runs the consumer.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    consumer = CortexConsumer()

    # Set up signal handlers for graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {sig}")
        consumer.stop()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Starting Cortex Consumer...")
        loop.run_until_complete(consumer.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        # Clean up the event loop
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            loop.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("Cortex Consumer shutdown complete")


if __name__ == "__main__":
    main()
