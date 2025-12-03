"""
Consumer Entry Point

Run the Cortex consumer with graceful shutdown handling.

Usage:
    python -m consumer.run_consumer
"""

import asyncio
import signal
import sys

from consumer.cortex import create_consumer, logger


def main():
    """
    Main entry point for the Cortex consumer.

    Sets up signal handlers for graceful shutdown and runs the consumer.
    """
    consumer = create_consumer()

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
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            # Wait for all tasks to complete
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
