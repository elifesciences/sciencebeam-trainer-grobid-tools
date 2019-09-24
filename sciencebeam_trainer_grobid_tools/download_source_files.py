import argparse
import logging
import os
from shutil import copyfileobj
from multiprocessing.dummy import Pool as ThreadPool
from typing import List, Tuple

from apache_beam.io.filesystems import FileSystems

from sciencebeam_utils.utils.file_list import (
    load_file_list,
    to_relative_file_list,
    to_absolute_file_list
)

from sciencebeam_utils.tools.check_file_list import map_file_list_to_file_exists


LOGGER = logging.getLogger(__name__)


DEFAULT_DOCUMENT_COLUMN = 'source_url'
DEFAULT_TARGET_COLUMN = 'xml_url'

DEFAULT_OUTPUT_FILENAME_PATTERN = '{filename}'


def _add_file_list_args(parser, name, label, default_file_column):
    parser.add_argument(
        '--%s-file-list' % name, type=str, required=True,
        help='path to %s file list (csv/tsv/lst)' % label
    )
    parser.add_argument(
        '--%s-base-path' % name, type=str, required=False,
        help='base path of %s file list (default is dirname of file list)' % label
    )
    parser.add_argument(
        '--%s-file-column' % name, type=str, required=False,
        default=default_file_column,
        help='csv/tsv column name (ignored for plain file list)'
    )


def get_file_list_config(args, name):
    return {
        'file_list': getattr(args, '%s_file_list' % name),
        'base_path': getattr(args, '%s_base_path' % name),
        'file_column': getattr(args, '%s_file_column' % name)
    }


def _add_output_args(parser, name, label):
    parser.add_argument(
        '--%s-output-path' % name, type=str, required=True,
        help='output path for %s' % label
    )
    parser.add_argument(
        '--%s-output-filename-pattern' % name, type=str, required=False,
        default=DEFAULT_OUTPUT_FILENAME_PATTERN,
        help='output filename of %s (defaults to source filename)' % label
    )


def get_output_config(args, name):
    return {
        'output_path': getattr(args, '%s_output_path' % name),
        'output_filename_pattern': getattr(args, '%s_output_filename_pattern' % name)
    }


def add_main_args(parser):
    source_group = parser.add_argument_group('source')
    document_source_group = source_group.add_argument_group('PDF document')
    target_source_group = source_group.add_argument_group('target XML')

    _add_file_list_args(document_source_group, 'document', 'PDF document', DEFAULT_DOCUMENT_COLUMN)
    _add_file_list_args(target_source_group, 'target', 'target XML', DEFAULT_TARGET_COLUMN)

    output_group = parser.add_argument_group('output')
    document_output_group = output_group.add_argument_group('PDF document')
    target_output_group = output_group.add_argument_group('target XML')

    _add_output_args(document_output_group, 'document', 'PDF document')
    _add_output_args(target_output_group, 'target', 'target XML')

    parser.add_argument(
        '--limit', type=int, required=False,
        help='limit the number of files to process'
    )

    parser.add_argument(
        '--threads', type=int, default=1,
        help='enable multi-threading (with the specified number of threads)'
    )

    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='enable debug output'
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_main_args(parser)

    parsed_args = parser.parse_args(argv)

    LOGGER.info('parsed_args: %s', parsed_args)

    return parsed_args


class FileList(object):
    def __init__(self, base_path, file_list):
        self.base_path = base_path
        if base_path:
            self.relative_file_list = to_relative_file_list(base_path, file_list)
            self.absolute_file_list = to_absolute_file_list(base_path, file_list)
        else:
            self.relative_file_list = file_list
            self.absolute_file_list = file_list

    def select(self, selection: List[bool]) -> 'FileList':
        assert len(selection) == len(self.absolute_file_list)
        return FileList(
            base_path=self.base_path,
            file_list=[
                file_path
                for file_path, selected in zip(self.absolute_file_list, selection)
                if selected
            ]
        )

    def __len__(self):
        return len(self.absolute_file_list)

    def __bool__(self):
        return len(self) > 0

    def __eq__(self, other: 'FileList') -> bool:
        if not isinstance(other, FileList):
            return False
        return (
            (self.base_path == other.base_path)
            and (self.relative_file_list == other.relative_file_list)
            and (self.absolute_file_list == other.absolute_file_list)
        )

    def __repr__(self):
        return '%s(base_path="%s", file_list=%s)' % (
            type(self).__name__, self.base_path, self.relative_file_list
        )


def load_file_list_from_config(file_list_config, limit):
    return FileList(
        base_path=file_list_config['base_path'],
        file_list=load_file_list(
            file_list_config['file_list'],
            column=file_list_config['file_column'],
            limit=limit
        )
    )


def get_filename_pattern_props(relative_source_filename):
    relative_dirname = os.path.dirname(relative_source_filename)
    filename = os.path.basename(relative_source_filename)
    name, ext = os.path.splitext(filename)
    if ext.lower() == '.gz':
        name, ext = os.path.splitext(name)
    return dict(
        dir=relative_dirname + '/' if relative_dirname else '',
        filename=filename,
        name=name,
        ext=ext
    )


def get_relative_output_filename(
        relative_source_filename,
        output_filename_pattern,
        index,
        file_lists=None):
    pattern_props = get_filename_pattern_props(relative_source_filename)
    for file_list_name, file_list in (file_lists or {}).items():
        pattern_props[file_list_name] = argparse.Namespace(
            **get_filename_pattern_props(file_list[index])
        )
    return output_filename_pattern.format(index=index, **pattern_props)


def get_relative_output_file_list(
        relative_source_file_list,
        output_filename_pattern,
        file_lists=None):
    return [
        get_relative_output_filename(
            relative_source_filename=filename,
            output_filename_pattern=output_filename_pattern,
            index=index,
            file_lists=file_lists
        )
        for index, filename in enumerate(relative_source_file_list)
    ]


def get_output_file_list(
        source_file_list,
        output_path,
        output_filename_pattern,
        file_lists=None):
    return FileList(
        base_path=output_path,
        file_list=[
            os.path.join(output_path, relative_output_path)
            for relative_output_path in get_relative_output_file_list(
                relative_source_file_list=source_file_list.relative_file_list,
                output_filename_pattern=output_filename_pattern,
                file_lists=file_lists
            )
        ]
    )


def get_output_file_list_from_config(
        source_file_list,
        output_config,
        file_lists=None):
    return get_output_file_list(
        source_file_list=source_file_list,
        output_path=output_config['output_path'],
        output_filename_pattern=output_config['output_filename_pattern'],
        file_lists=file_lists
    )


def mkdirs_if_not_exists(path):
    try:
        FileSystems.mkdirs(path)
    except IOError:
        pass


def copy_file(source_filename, output_filename):
    LOGGER.debug(
        'copy_file: source_filename=%s, output_filename=%s',
        source_filename, output_filename
    )
    with FileSystems.open(source_filename) as source_fp:
        mkdirs_if_not_exists(os.path.dirname(output_filename))
        with FileSystems.create(output_filename) as output_fp:
            copyfileobj(source_fp, output_fp)


def copy_files(source_file_list, output_file_list, pool=None):  # pylint: disable=unused-argument
    LOGGER.debug(
        'copy_files: source_file_list=%s, output_file_list=%s',
        source_file_list, output_file_list
    )
    assert len(source_file_list) == len(output_file_list)
    if pool:
        pool.map(lambda args: copy_file(*args), zip(source_file_list, output_file_list))
    else:
        for source_filename, output_filename in zip(source_file_list, output_file_list):
            copy_file(source_filename, output_filename)


def filter_file_pair_exists(
        file_list1: FileList, file_list2: FileList) -> Tuple[FileList, FileList]:
    file_list_exists1 = map_file_list_to_file_exists(
        file_list1.absolute_file_list
    )
    file_list_exists2 = map_file_list_to_file_exists(
        file_list2.absolute_file_list
    )
    pair_exists = [
        exists1 and exists2
        for exists1, exists2 in zip(file_list_exists1, file_list_exists2)
    ]
    return (
        file_list1.select(pair_exists),
        file_list2.select(pair_exists)
    )


def run(args):
    all_document_file_list = load_file_list_from_config(
        get_file_list_config(args, 'document'),
        limit=args.limit
    )
    all_target_file_list = load_file_list_from_config(
        get_file_list_config(args, 'target'),
        limit=args.limit
    )

    LOGGER.debug('all_document_file_list: %s', all_document_file_list)
    LOGGER.debug('all_target_file_list: %s', all_target_file_list)

    document_file_list, target_file_list = filter_file_pair_exists(
        all_document_file_list, all_target_file_list
    )

    if not document_file_list:
        raise ValueError('none of the file pairs exists')

    if len(document_file_list) < len(all_document_file_list):
        LOGGER.warning(
            'not all file pair exists: %d exists (total: %d)',
            len(document_file_list), len(all_document_file_list)
        )

    file_lists = {
        'document': document_file_list.relative_file_list,
        'target': target_file_list.relative_file_list
    }

    document_output_file_list = get_output_file_list_from_config(
        source_file_list=document_file_list,
        output_config=get_output_config(args, 'document'),
        file_lists=file_lists
    )

    target_output_file_list = get_output_file_list_from_config(
        source_file_list=target_file_list,
        output_config=get_output_config(args, 'target'),
        file_lists=file_lists
    )

    LOGGER.debug('document_output_file_list: %s', document_output_file_list)
    LOGGER.debug('target_output_file_list: %s', target_output_file_list)

    pool = ThreadPool(args.threads)

    copy_files(
        document_file_list.absolute_file_list,
        document_output_file_list.absolute_file_list,
        pool=pool
    )

    copy_files(
        target_file_list.absolute_file_list,
        target_output_file_list.absolute_file_list,
        pool=pool
    )

    pool.close()
    pool.join()


def main(argv=None):
    args = parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel('DEBUG')

    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
