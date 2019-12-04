from collections import defaultdict
from collections import deque
import time
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
    IMPORTANT = 25
    WARN = 30
    ERROR = 40
    DISABLED = 50

    def __init__(self):

        self.output_log = []
        self.print_level = self.INFO

        self.csv_path = None
        self.txt_path = None

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

    def print_variables(self, include_header=False):
        """ Prints current value of all logged variables."""

        sorted_vars = sorted(self._vars.values())

        if include_header:
            output_string = ""
            for var in sorted_vars:
                if var.display_width == 0:
                    continue
                output_string = output_string + (var.display_name.rjust(var.display_width-1, " ")+" ")[:var.display_width]
            self.log("-" * len(output_string))
            self.log(output_string)
            self.log("-"*len(output_string))

        output_string = ""
        for var in sorted_vars:
            if var.display_width == 0:
                continue
            output_string = output_string + (var.display.rjust(var.display_width - 1, " ") + " ")
        self.log(output_string)

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

    def log(self, s="", level=INFO):
        s =  str(s)
        if level >= self.print_level:
            if level == self.IMPORTANT:
                s = "<green>{}<end>".format(s)
            elif level == self.WARN:
                s = "<yellow>{}<end>".format(s)
            elif level == self.ERROR:
                s = "<red>{}<end>".format(s)
        print(color_format_string(s))
        self.output_log.append((level, time.time(), color_format_string(s, strip_colors=True)))

    def debug(self, s=""):
        self.log(s, level=self.DEBUG)

    def info(self, s=""):
        self.log(s, level=self.INFO)

    def important(self, s=""):
        self.log(s, level=self.IMPORTANT)

    def warn(self, s=""):
        self.log(s, level=self.WARN)

    def error(self, s=""):
        self.log(s, level=self.ERROR)

    def export_to_csv(self, file_name=None):

        if len(self._history) == 0:
            return

        file_name = file_name or self.csv_path

        with open(file_name, "w") as f:
            field_names = self._history[-1].keys()
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for row in self._history:
                writer.writerow(row)

    def save_log(self, file_name=None):
        file_name = file_name or self.txt_path
        # note it would be better to simply append the new lines?
        with open(file_name, "w") as f:
            lines = ["[{:<10}] {:<20} {}".format(level, str(time), line) for level, time, line in self.output_log]
            f.writelines(lines)

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


def color_format_string(s, strip_colors=False):
    """ Converts color tags into color strings, or removes them."""
    color_table = {
        "<red>": utils.Color.FAIL,
        "<green>": utils.Color.OKGREEN,
        "<white>": utils.Color.BOLD,
        "<blue>": utils.Color.OKBLUE,
        "<purple>": utils.Color.HEADER,
        "<yellow>": utils.Color.WARNING,
        "<end>": utils.Color.ENDC
    }

    for k,v in color_table:
        if strip_colors:
            v = ""
        s = s.replace(k, v)
    return s

def nice_round(x, rounding):

    if rounding == 0:
        return round(x, 1) if abs(x) < 100 else "{:,}".format(int(x))

    if rounding > 0:
        return round(x, rounding)
