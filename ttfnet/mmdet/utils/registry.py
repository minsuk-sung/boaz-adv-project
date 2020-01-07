import inspect

import mmcv


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))

        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        # print('obj_type_before:',obj_type)
        obj_type = registry.get(obj_type)
        # print('obj_type_after:',obj_type)
        # obj_type_before: TTFNet
        # obj_type_after: <class 'mmdet.models.detectors.ttfnet.TTFNet'>
        # obj_type_before: DarknetV3
        # obj_type_after: <class 'mmdet.models.backbones.darknet.DarknetV3'>

        if obj_type is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            # name: train_cfg value: {'vis_every_n_iters': 100, 'debug': False}
            # name: test_cfg value: {'score_thr': 0.01, 'max_per_img': 100
            args.setdefault(name, value)
    # print('obj_type:',obj_type(**args))
    return obj_type(**args)
