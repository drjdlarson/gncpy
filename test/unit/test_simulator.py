import pytest

from gncpy.simulator import Parameters


class TempChildClass:
    def __init__(self):
        self.c_int = 0
        self.c_d = {}


class TempClass:
    def __init__(self):
        self.var = 0
        self.lst = []
        self.d = {}
        self.c = TempChildClass()


@pytest.mark.incremental
class TestParameters:
    def test_load_files(self, yaml_file, yaml_file_lst):
        params = Parameters()
        params.load_files(yaml_file)
        assert params.dict['tempClass']['var'] == 5
        assert params.dict['tempClass']['lst'][0] == 'str'
        assert params.dict['tempClass']['c']['c_int'] == 7

        params = Parameters()
        params.load_files(yaml_file_lst)
        assert params.dict['tempClass']['var'] == 10
        assert params.dict['tempClass']['c']['c_int'] == 7

    def test_init_class(self, yaml_file):
        params = Parameters()
        params.load_files(yaml_file)

        test_cls = TempClass()
        params.init_class(test_cls, 'tempClass')
        assert test_cls.var == 5
        assert test_cls.lst[1] == 0
        assert test_cls.c.c_d['key4'] == 4
        assert test_cls.c.c_int == 7

    def test_save_config(self):
        assert 0
