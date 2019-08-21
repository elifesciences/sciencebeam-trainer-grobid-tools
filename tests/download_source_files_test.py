import os
from pathlib import Path
from unittest.mock import patch, call

import pytest

from sciencebeam_utils.utils.file_list import save_file_list

import sciencebeam_trainer_grobid_tools.download_source_files as download_source_files_module
from sciencebeam_trainer_grobid_tools.download_source_files import (
    get_relative_output_file_list,
    get_output_file_list,
    FileList,
    copy_files,
    main,
    DEFAULT_DOCUMENT_COLUMN,
    DEFAULT_TARGET_COLUMN
)


@pytest.fixture(name='load_file_list_mock')
def _load_file_list_mock():
    with patch.object(download_source_files_module, 'load_file_list') as mock:
        yield mock


@pytest.fixture(name='load_file_list_from_config_mock')
def _load_file_list_from_config_mock():
    with patch.object(download_source_files_module, 'load_file_list_from_config') as mock:
        yield mock


@pytest.fixture(name='copy_files_mock')
def _copy_files_mock():
    with patch.object(download_source_files_module, 'copy_files') as mock:
        yield mock


def get_default_args(base_path):
    return {
        'document_file_list': os.path.join(base_path, 'source/document-file-list.lst'),
        'target_file_list': os.path.join(base_path, 'source/target-file-list.lst'),
        'document_output_path': os.path.join(base_path, 'output/document'),
        'target_output_path': os.path.join(base_path, 'output/target')
    }


def to_argv(args_dict):
    argv = []
    for key, value in args_dict.items():
        arg_name = '--%s' % key.replace('_', '-')
        argv.append(arg_name)
        argv.append(value)
    return argv


class TestGetRelativeOutputFileList(object):
    def test_should_return_empty_list_if_source_file_list_is_empty(self):
        assert get_relative_output_file_list(
            [],
            output_filename_pattern=''
        ) == []

    def test_should_use_source_filename(self):
        source_file_list = ['file1.pdf']
        assert get_relative_output_file_list(
            source_file_list,
            output_filename_pattern='{filename}'
        ) == ['file1.pdf']

    def test_should_use_index_and_source_ext(self):
        source_file_list = ['file1.pdf']
        assert get_relative_output_file_list(
            source_file_list,
            output_filename_pattern='index-{index}{ext}'
        ) == ['index-0.pdf']

    def test_should_not_use_source_relative_subdirectory_if_not_in_pattern(self):
        source_file_list = ['sub/file1.pdf']
        assert get_relative_output_file_list(
            source_file_list,
            output_filename_pattern='{filename}'
        ) == ['file1.pdf']

    def test_should_use_source_relative_subdirectory_if_in_pattern(self):
        source_file_list = ['sub/file1.pdf']
        assert get_relative_output_file_list(
            source_file_list,
            output_filename_pattern='{dir}{filename}'
        ) == ['sub/file1.pdf']

    def test_should_use_ref1_name_and_source_ext(self):
        source_file_list = ['file1.pdf']
        ref_file_list = ['ref1.xy']
        assert get_relative_output_file_list(
            source_file_list,
            output_filename_pattern='{ref.name}{ext}',
            file_lists={
                'ref': ref_file_list
            }
        ) == ['ref1.pdf']


class TestGetOutputFileList(object):
    def test_should_return_empty_list_if_source_file_list_is_empty(self):
        assert get_output_file_list(
            source_file_list=FileList(base_path='/source', file_list=[]),
            output_path='/output',
            output_filename_pattern=''
        ).absolute_file_list == []

    def test_should_map_source_path_to_output_path(self):
        assert get_output_file_list(
            source_file_list=FileList(base_path='/source', file_list=['/source/sub/file1.pdf']),
            output_path='/output',
            output_filename_pattern='{dir}{filename}'
        ).absolute_file_list == ['/output/sub/file1.pdf']

    def test_should_map_ref_path_to_output_path(self):
        assert get_output_file_list(
            source_file_list=FileList(base_path='/source', file_list=['/source/sub/file1.pdf']),
            output_path='/output',
            output_filename_pattern='{ref.dir}{ref.filename}',
            file_lists={
                'ref': ['refsub/reffile1.pdf']
            }
        ).absolute_file_list == ['/output/refsub/reffile1.pdf']


class TestCopyFiles(object):
    def test_should_copy_single_file_and_create_output_directory(self, tmpdir):
        source_file = Path(tmpdir).joinpath('source').joinpath('file1.pdf')
        output_file = Path(tmpdir).joinpath('output').joinpath('file1.pdf')
        source_file.parent.mkdir()
        source_file.write_bytes(b'file1.pdf content')
        copy_files([str(source_file)], [str(output_file)])
        assert output_file.read_bytes()

    def test_should_copy_single_file_and_use_existing_output_directory(self, tmpdir):
        source_file = Path(tmpdir).joinpath('source').joinpath('file1.pdf')
        output_file = Path(tmpdir).joinpath('output').joinpath('file1.pdf')
        source_file.parent.mkdir()
        output_file.parent.mkdir()
        source_file.write_bytes(b'file1.pdf content')
        copy_files([str(source_file)], [str(output_file)])
        assert output_file.read_bytes()

    def test_should_copy_files_using_pool(self, tmpdir, thread_pool):
        source_file = Path(tmpdir).joinpath('source').joinpath('file1.pdf')
        output_file = Path(tmpdir).joinpath('output').joinpath('file1.pdf')
        source_file.parent.mkdir()
        source_file.write_bytes(b'file1.pdf content')
        copy_files([str(source_file)], [str(output_file)], pool=thread_pool)
        assert output_file.read_bytes()


class TestMain(object):
    @pytest.mark.usefixtures('copy_files_mock')
    def test_should_call_load_file_list(
            self, tmpdir, load_file_list_from_config_mock):
        args = get_default_args(str(tmpdir))
        args['limit'] = '123'
        main(to_argv(args))
        load_file_list_from_config_mock.assert_has_calls([call({
            'file_list': args['document_file_list'],
            'base_path': args.get('document_base_path'),
            'file_column': args.get('document_file_column', DEFAULT_DOCUMENT_COLUMN),
        }, limit=123), call({
            'file_list': args['target_file_list'],
            'base_path': args.get('target_base_path'),
            'file_column': args.get('target_file_column', DEFAULT_TARGET_COLUMN)
        }, limit=123)], any_order=True)

    def test_should_copy_files(self, tmpdir):
        args = get_default_args(str(tmpdir))
        document_file_list_path = Path(args['document_file_list'])
        document_file_list_path.parent.mkdir(exist_ok=True)
        save_file_list(
            str(document_file_list_path),
            column=args.get('document_file_column'),
            file_list=['file1.pdf']
        )
        target_file_list_path = Path(args['target_file_list'])
        target_file_list_path.parent.mkdir(exist_ok=True)
        save_file_list(
            str(target_file_list_path),
            column=args.get('target_file_column'),
            file_list=['file1.xml']
        )
        Path(args['document_file_list']).parent.joinpath('file1.pdf').write_bytes(
            b'file1.pdf content'
        )
        Path(args['target_file_list']).parent.joinpath('file1.xml').write_bytes(
            b'file1.xml content'
        )
        main(to_argv(args))

        assert (
            Path(args['document_output_path']).joinpath('file1.pdf').read_bytes()
            == b'file1.pdf content'
        )

        assert (
            Path(args['target_output_path']).joinpath('file1.xml').read_bytes()
            == b'file1.xml content'
        )
