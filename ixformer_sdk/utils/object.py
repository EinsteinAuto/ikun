import inspect
from typing import Any, Callable, Dict, Mapping, Union

__all__ = [
    "isfunction",
    "iscallable",
    "get_obj_name",
    "isimmutable_var",
    "get_self_from",
    "get_obj_funcs",
    "recurse_getattr",
    "recurse_find_by_key",
    "set_value_by_cascasde_key",
    "flatten_container",
    "flatten_dict",
    "get_func_argspec",
    "get_obj_attr",
    "get_namedtuple_fields",
    "get_namedtuple_defaults",
    "isnamedtuple",
    "namedtype_to_dict",
]


def isfunction(f):
    return (
        inspect.isfunction(f) or inspect.ismethod(f) or inspect.isbuiltin(f)
    ) and not inspect.isclass(f)


def iscallable(fn) -> bool:
    return any(
        [
            callable(fn),
            inspect.isfunction(fn),
            inspect.ismethod(fn),
            inspect.isbuiltin(fn),
        ]
    )


def get_obj_name(obj, containe_module=False):
    mod_name = None
    if inspect.isclass(obj):
        obj_name = obj.__name__
        if hasattr(obj, "__module__") and containe_module:
            mod_name = obj.__module__
    elif hasattr(obj, "__name__"):
        obj_name = obj.__name__
    elif hasattr(obj, "__class__"):
        obj_name = obj.__class__.__name__
        if hasattr(obj.__class__, "__module__") and containe_module:
            mod_name = obj.__class__.__module__
    else:
        obj_name = str(obj)

    if containe_module and mod_name is None:
        if hasattr(obj, "__module__"):
            mod_name = obj.__module__

    if mod_name is None:
        return obj_name
    else:
        return f"{mod_name}.{obj_name}"


def isimmutable_var(var):
    if var is None:
        return True

    if inspect.isclass(var):
        var_cls = var
    else:
        var_cls = type(var)

    return var_cls in [int, float, tuple, str, None]


def get_self_from(obj):
    if hasattr(obj, "__self__"):
        return obj.__self__
    raise AttributeError(f"Not found attribute `self` in {obj}.")


def get_obj_funcs(obj) -> Dict[str, Callable]:
    attrs = dir(obj)
    funcs = dict()
    for attr in attrs:
        fn = getattr(obj, attr)
        if iscallable(fn):
            funcs[attr] = fn

    return funcs


def recurse_find_by_key(container: dict, key: Union[str, list], default=None):
    if isinstance(key, str):
        key = key.split(".")

    if not isinstance(key, (tuple, list)):
        raise RuntimeError(f"Please give the type str or list, but get ({type(key)}).")

    value = default
    _cnt = container
    for k in key:
        if k not in _cnt:
            return default
        value = _cnt[k]
        _cnt = value
        if _cnt is None:
            return default

    return value


def set_value_by_cascasde_key(container: dict, key: str, value: Any):
    if isinstance(key, str):
        key = key.split(".")

    if not isinstance(key, (tuple, list)):
        raise RuntimeError(f"Please give the type str or list, but get ({type(key)}).")

    _cnt = container
    for k in key[:-1]:
        if k not in _cnt:
            _cnt[k] = dict()
        _cnt = _cnt[k]
    _cnt[key[-1]] = value
    return container


def flatten_dict(d: dict, preffix="", out=None):
    if out is None:
        out = dict()
    for k, v in d.items():
        if isinstance(v, Mapping):
            flatten_dict(v, f"{preffix}{k}.", out)
        else:
            out[preffix + k] = v

    return out


def flatten_container(container: Union[list, dict]):
    outs = []

    def _flatten_list(cnt: list):
        for item in cnt:
            if isinstance(item, (tuple, list)):
                _flatten_list(item)
            elif isinstance(item, Mapping):
                _flatten_dict(item)
            else:
                outs.append(item)

    def _flatten_dict(cnt: Dict):
        for key, item in cnt.items():
            if isinstance(item, (tuple, list)):
                _flatten_list(item)
            elif isinstance(item, Mapping):
                _flatten_dict(item)
            else:
                outs.append(item)

    if isinstance(container, (tuple, list)):
        _flatten_list(container)
    elif isinstance(container, dict):
        _flatten_dict(container)
    else:
        outs.append(container)

    return outs


def get_func_argspec(func) -> inspect.FullArgSpec:
    return inspect.getfullargspec(func)


def get_obj_attr(obj, attr, default=None):
    if isinstance(obj, Mapping):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def isnamedtuple(obj):
    if not inspect.isclass(obj) or not issubclass(obj, tuple):
        return False

    if hasattr(obj, "_fields") and hasattr(obj, "_replace"):
        if (
            hasattr(obj._replace, "__module__")
            and obj._replace.__module__ == "collections"
        ):
            return True

    return False


def get_namedtuple_fields(t):
    if not inspect.isclass(t):
        t = type(t)

    if not isnamedtuple(t):
        raise RuntimeError(f"{t} is not a namedtuple object")

    return t._fields


def get_namedtuple_defaults(t) -> dict:
    return t._field_defaults


def namedtype_to_dict(t):
    return t._asdict()


def recurse_getattr(obj, attr: str, sep="."):
    attrs = attr.split(sep)
    idx = 0
    cur_obj = obj
    while idx < len(attrs):
        cur_obj = getattr(cur_obj, attrs[idx])
        idx += 1

    if cur_obj == obj:
        return None
    return cur_obj
