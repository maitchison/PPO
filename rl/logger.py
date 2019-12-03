from collections import defaultdict
from collections import deque
from . import utils

import numpy as np
import csv
import torch

class LogVariable():
    """
    Logs a variable with support for querying its mean, std, min and max over some fixed timeframe.
    """

    def __init__(self, name, history_length=1, type="float", display_width=None, display_precision=None,
                 display_priority=0, export_precision=None, display_postfix="", display_scale=1, display_name=None):

        assert type in ["int", "float", "stats", "str"]

        default_display_width = {
            "int": 10,
            "float": 10,
            "stats": 26,
            "str": 20
        }[type]

        default_display_precision = {
            "int": 0,
            "float": 3,
            "stats": 0,
            "str": None
        }[type]

        default_export_precision = {
            "int": 1,
            "float": 6,
            "stats": 6,
            "str": None
        }[type]

        self._name = name
        self._display_name = display_name
        self.display_width = utils.default(display_width, max(default_display_width, len(self.display_name) + 1))
        self.display_precision = utils.default(display_precision, default_display_precision)
        self.export_precision = utils.default(export_precision, default_export_precision)
        self.display_priority = display_priority
        self.display_postfix = display_postfix
        self.display_scale = display_scale

        self._history = deque(maxlen=history_length)
        self._history_length = history_length
        self._type = type

    def add_sample(self, value):

        # cast input into the correct type.
        if self._type == "int":
            value = int(value)
        elif self._type == "float":
            value = float(value)
        elif self._type == "stats":
            value = float(value)
        elif self._type == "str":
            value = str(value)

        self._history.append(value)

    def get_history(self, max_samples=None):
        if max_samples is None:
            start = None
        else:
            start = -max_samples
        return self._history[start:]

    @property
    def name(self):
        return self._name

    @property
    def _sort_key(self):
        return (-self.display_priority, self.name)

    def __lt__(self, other):
        return self._sort_key < other._sort_key

    @property
    def display_name(self):
        return self.name if self._display_name is None else self._display_name

    @property
    def display(self):
        """ Returns formatted value. """

        value = self.value
        if self._type == "int":
            result = ("{:,."+str(self.display_precision)+"f}").format(value*self.display_scale)
        elif self._type == "float":
            result = str(nice_round(value*self.display_scale, self.display_precision))
        elif self._type == "stats":
            result = "{} ±{} ({}/{})".format(*(nice_round(x, self.display_precision) for x in value))
        elif self._type == "str":
            result = str(value)
        else:
            raise Exception("Invalid type {} for log variable.".format(self._type))

        return (result+self.display_postfix)[:(self.display_width-1)]

    @property
    def value(self):
        if self._type == "stats":
            return tuple(nice_round(func(self._history), self.export_precision) if len(self._history) > 0 else 0 for func in [np.mean, np.std, np.min, np.max])
        elif self._type == "str":
            return self._history[-1] if len(self._history) > 0 else ""
        elif self._type == "float":
            return nice_round(float(np.mean(self._history)), self.export_precision) if len(self._history) > 0 else 0
        elif self._type == "int":
            return nice_round(float(np.mean(self._history)), self.export_precision) if len(self._history) > 0 else 0

class Logger():
    """
        Class to handle training logging.
    """

    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    DISABLED = 50

    def __init__(self):

        self.output_log = []
        self.print_level = self.INFO

        self._vars = {}
        self._history = []

    def add_variable(self, variable: LogVariable):
        self._vars[variable.name] = variable

    def watch(self, key, value, **kwargs):
        """ Logs a value, creates log variable if needed. """
        if key not in self._vars:
            # work out which type to use.
            if "type" not in kwargs:
                kwargs["type"] = assume_type(value)
            self.add_variable(LogVariable(key, **kwargs))
        self._vars[key].add_sample(value)

    def watch_mean(self, key, value, history_length=10, **kwargs):
        """ Logs a value, creates mean log variable if needed. """
        self.watch(key, value, history_length=history_length, **kwargs)

    def watch_full(self, key, value, history_length=100, **kwargs):
        """ Logs a value, creates full variable if needed. """
        self.watch(key, value, history_length=history_length, type="stats", **kwargs)

    def print(self, include_header=False):
        """ Prints current value of all logged variables."""

        sorted_vars = sorted(self._vars.values())

        if include_header:
            output_string = ""
            for var in sorted_vars:
                if var.display_width == 0:
                    continue
                output_string = output_string + (var.display_name.rjust(var.display_width-1, " ")+" ")[:var.display_width]
            print("-" * len(output_string))
            print(output_string)
            print("-"*len(output_string))

        output_string = ""
        for var in sorted_vars:
            if var.display_width == 0:
                continue
            output_string = output_string + (var.display.rjust(var.display_width - 1, " ") + " ")
        print(output_string)

    def record_step(self):
        """ Records state of all watched variables for this given step. """
        row = {}
        for var in sorted(self._vars.values()):
            if var._type == "stats":
                row[var.name + "_mean"] = var.value[0]
                row[var.name + "_std"] = var.value[1]
                row[var.name + "_min"] = var.value[2]
                row[var.name + "_max"] = var.value[3]
            else:
                row[var.name] = var.value
        self._history.append(row)

    def log(self, s, level=INFO):
        if level >= self.print_level:
            print(s)
        self.output_log.append((level, s))

    def debug(self, s):
        self.log(s, level=self.DEBUG)

    def info(self, s):
        self.log(s, level=self.INFO)

    def warn(self, s):
        self.log(s, level=self.WARN)

    def error(self, s):
        self.log(s, level=self.ERROR)

    def export_to_csv(self, file_name):

        if len(self._history) == 0:
            return

        with open(file_name, "w") as f:
            field_names = self._history[-1].keys()
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for row in self._history:
                writer.writerow(row)

    def export_to_tensor_board(self):
        raise NotImplemented()

    def export_to_graphs(self):
        raise NotImplemented()

    def __getitem__(self, key):
        return self._vars[key].value

def assume_type(value):
    """ Returns the type, int, float or str of variable. Should work fine with np variables. """

    if type(value) == torch.Tensor:
        assert len(value.shape) == 0, "Torch tensor must be scalar, but found shape {}".format(value.shape)
        if value.dtype == torch.float:
            return "float"
        if value.dtype == torch.int:
            return "int"
        raise Exception("Can not infer type {} with dtype {}.".format(type(value), value.dtype))

    if type(value) == str:
        return "str"

    if type(value) == int:
        return "int"

    if type(value) == float:
        return "float"

    if np.issubdtype(type(value), np.integer):
        return "int"

    if np.issubdtype(type(value), np.floating):
        return "float"

    raise Exception("Can not infer type {}.".format(type(value)))


def nice_round(x, rounding):

    if rounding == 0:
        return round(x, 1) if abs(x) < 100 else "{:,}".format(int(x))

    if rounding > 0:
        return round(x, rounding)
