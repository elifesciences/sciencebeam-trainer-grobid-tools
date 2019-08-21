import logging
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def configure_logging():
    logging.root.handlers = []
    logging.basicConfig(level='DEBUG')


@pytest.fixture()
def thread_pool():
    pool = ThreadPool(5)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


@pytest.fixture()
def temp_dir(tmpdir):
    return Path(tmpdir)
