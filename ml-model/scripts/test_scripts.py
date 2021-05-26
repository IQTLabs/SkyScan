'''test all scripts.'''

from main import read_config


def test_read_config():
    '''Test read_config().'''
    config = read_config('test_config.ini')
    assert config['file_names']['dataset_name'] == 'hello_world'
    assert config['file_locations']['image_directory'] == 'foo'