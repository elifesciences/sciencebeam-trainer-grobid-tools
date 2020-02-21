import logging
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def configure_logging():
    logging.root.setLevel('INFO')
    for name in ['tests', 'sciencebeam_trainer_grobid_tools']:
        logging.getLogger(name).setLevel('DEBUG')


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
