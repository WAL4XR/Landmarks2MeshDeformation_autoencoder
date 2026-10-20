"""API entry point for model serving."""

import yaml
import logging
from typing import Any

# from fastapi import FastAPI
# app = FastAPI()

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/deploy_config.yaml") -> dict:
    """Load deployment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_app():
    """Create and configure the API application."""
    config = load_config()
    logger.info(f"Starting service on {config['server']['host']}:{config['server']['port']}")
    # TODO: Initialize FastAPI/Flask app and load model
    raise NotImplementedError("Service not yet implemented.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app()
