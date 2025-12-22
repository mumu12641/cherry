import functools
import json


class CherryJSONLoader:
    def __init__(self, ir, config_path):
        self.ir = ir
        with open(config_path, "r") as f:
            self.data = json.load(f)

        self.constants = self.data.get("config", {})
        self.type_cache = {}

    def _resolve_dim(self, dim):
        if isinstance(dim, str):
            if dim not in self.constants:
                raise ValueError(f"Undefined constant dimension: {dim}")
            return self.constants[dim]
        return dim

    def _create_mlir_type(self, type_def):
        if isinstance(type_def, str):
            if type_def not in self.type_cache:
                self.type_cache[type_def] = self.ir.create_type(type_def)
            return self.type_cache[type_def]

        if isinstance(type_def, list) and len(type_def) == 2:
            shape_cfg = type_def[0]
            dtype_str = type_def[1]

            resolved_shape = [self._resolve_dim(d) for d in shape_cfg]

            elem_type = self._create_mlir_type(dtype_str)

            return self.ir.create_tensor_type(resolved_shape, elem_type)

        raise ValueError(f"Unknown type definition: {type_def}")


def from_json_config(ir, json_path, func_key=None):
    loader = CherryJSONLoader(ir, json_path)

    def decorator(func):
        target_key = func_key if func_key else func.__name__

        if target_key not in loader.data["functions"]:
            raise KeyError(f"Function '{target_key}' not found in {json_path}")

        func_config = loader.data["functions"][target_key]

        mlir_arg_types = []
        for input_def in func_config["inputs"]:
            mlir_arg_types.append(loader._create_mlir_type(input_def["type"]))

        mlir_ret_types = []
        for output_def in func_config["outputs"]:
            mlir_ret_types.append(loader._create_mlir_type(output_def["type"]))

        is_private = func_config.get("private", False)

        arg_values = ir.create_function(
            target_key, mlir_arg_types, mlir_ret_types, is_private
        )

        results = func(*arg_values)

        if results is None:
            results = []
        elif not isinstance(results, (list, tuple)):
            results = [results]

        ir.ret(list(results))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Function {target_key} generated from JSON config.")
        return wrapper

    return decorator
