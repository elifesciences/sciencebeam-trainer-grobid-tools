import os
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Union

from apache_beam.io.filesystems import FileSystems


@contextmanager
def auto_download_input_file(file_url_or_open_fn: Union[str, callable]) -> str:
    if isinstance(file_url_or_open_fn, Path):
        file_url_or_open_fn = str(file_url_or_open_fn)
    if isinstance(file_url_or_open_fn, str) and '://' not in file_url_or_open_fn:
        yield file_url_or_open_fn
        return
    with TemporaryDirectory(suffix='-input') as temp_dir:
        if isinstance(file_url_or_open_fn, str):
            temp_file = os.path.join(temp_dir, os.path.basename(file_url_or_open_fn))
            FileSystems.copy(file_url_or_open_fn, temp_file)
        else:
            temp_file = os.path.join(temp_dir, 'temp.file')
            with file_url_or_open_fn() as source:
                with open(temp_file, mode='wb') as temp_fp:
                    copyfileobj(source, temp_fp)
        yield temp_file
