"""
Main entry point for RefImage application.

This module provides the main function and CLI interface
for running the RefImage server.
"""

import logging

import uvicorn

from .api import create_app
from .config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Create app instance for uvicorn
try:
    settings = Settings()
    app = create_app(settings)
except Exception as e:
    logger.warning(f"Failed to create app instance: {e}")
    # Create minimal app for development
    from fastapi import FastAPI
    app = FastAPI(title="RefImage API (Development)")


def main():
    """Main application entry point."""
    try:
        # Load settings
        settings = Settings()

        # Create FastAPI app
        app = create_app(settings)

        # Update global app instance
        from . import api

        api.app = app

        # Run server
        logger.info("Starting RefImage server...")
        uvicorn.run(
            app,
            host=settings.server_host,
            port=settings.server_port,
            log_level=settings.log_level.lower(),
        )

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    main()
