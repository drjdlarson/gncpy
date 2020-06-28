import yaml
import io
import abc
import inspect
from warnings import warn


class Parameters:
    def __init__(self):
        self.dict = {}

    def load_files(self, files):
        def parse_file(self, f):
            with io.open(f, 'r') as fin:
                cfg = yaml.safe_load(fin)

            self.dict = merge(self.dict, cfg)

        def merge(base, new):
            for k, v in new.items():
                if isinstance(v, dict) and k in base:
                    base[k] = merge(base[k], v)
                else:
                    base[k] = v
            return base

        if isinstance(files, list) or isinstance(files, tuple):
            for f in files:
                parse_file(self, f)
        else:
            parse_file(self, files)

    def init_class(self, cls_obj, key_to_find, **kwargs):
        dct = kwargs.get('dct', self.dict)

        if key_to_find in dct:
            val = dct[key_to_find]

            if isinstance(val, dict) and not hasattr(cls_obj, key_to_find):
                for k, v in val.items():
                    if hasattr(cls_obj, k):
                        is_prim = isinstance(v, (str, int, float, list))
                        cond = not isinstance(getattr(cls_obj, k), dict) and \
                            not is_prim
                        if cond:
                            print(k)
                            if isinstance(v, dict):
                                self.init_class(getattr(cls_obj, k), k,
                                                dct=val)
                            else:
                                name = getattr(cls_obj, k).__class__.__name__
                                msg = "{} is a class ".format(name) \
                                    + "but key {} ".format(k) \
                                    + "is not a dictionary"
                                warn(msg, RuntimeWarning)
                        else:
                            setattr(cls_obj, k, v)
                    else:
                        name = cls_obj.__class__.__name__
                        msg = "Key {} not found in class {}".format(k, name)
                        warn(msg)
            elif hasattr(cls_obj, key_to_find):
                setattr(cls_obj, key_to_find, val)
            else:
                name = cls_obj.__class__.__name__
                msg = "Key {} not found in class {}".format(key_to_find, name)
                warn(msg, RuntimeWarning)
        else:
            msg = "Key {} not found in dictionary".format(key_to_find)
            warn(msg, RuntimeWarning)

    def save_config(self, file_name):
        with io.open(file_name, 'w') as fout:
            yaml.dump(self.dict, fout, default_flow_style=False)


class Simulation(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.sim_end_time = kwargs.get('sim_end_time', 0)
        self.base_time_step = kwargs.get('base_time_step', 0)

    def run(self, **kwargs):
        for kk in range(0, self.sim_end_time, self.base_time_step):
            self.iterate(**kwargs)

    @abc.abstractmethod
    def iterate(self, **kwargs):
        pass
