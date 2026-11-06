"""Run project tests with Python socket-connect auditing; no real network."""
import os
import sys

os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


def block_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("Real network access is forbidden in E08 offline tests")


sys.addaudithook(block_network)
import pytest

raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/"]))
