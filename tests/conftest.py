import sys
import types


def _stub_module(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


try:
    import anomalib.deploy  # noqa: F401
except ImportError:
    anomalib_mod = _stub_module("anomalib")
    deploy_mod = _stub_module("anomalib.deploy", OpenVINOInferencer=object)
    anomalib_mod.deploy = deploy_mod

try:
    import ultralytics  # noqa: F401
except ImportError:
    _stub_module("ultralytics", YOLO=object)

try:
    import mlflow  # noqa: F401
except ImportError:
    _stub_module("mlflow")
