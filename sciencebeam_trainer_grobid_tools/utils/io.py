import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import BinaryIO, Callable, Iterator, Union

from apache_beam.io.filesystems import FileSystems


T_BinaryIO_Open_Function = Callable[[], BinaryIO]


@contextmanager
def auto_download_input_file(
    file_url_or_open_fn: Union[str, T_BinaryIO_Open_Function]
) -> Iterator[str]:
    file_url = None
    open_fn = None
    if isinstance(file_url_or_open_fn, (Path, str)):
        file_url = str(file_url_or_open_fn)
    else:
        open_fn = file_url_or_open_fn

    if file_url and '://' not in file_url:
        yield file_url
        return
    with TemporaryDirectory(suffix='-input') as temp_dir:
        if file_url:
            temp_file = os.path.join(temp_dir, os.path.basename(file_url))
            open_fn = partial(FileSystems.open, file_url)
        else:
            temp_file = os.path.join(temp_dir, 'temp.file')
        with open_fn() as source:
            with open(temp_file, mode='wb') as temp_fp:
                copyfileobj(source, temp_fp)
        yield temp_file
