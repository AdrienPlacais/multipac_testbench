"""Provide data for testing purposes."""

from importlib import resources

dir = resources.files(__name__)
config_path = dir / "testbench_configuration.toml"
