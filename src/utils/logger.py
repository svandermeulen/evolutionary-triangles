"""
Written by: stef.vandermeulen
Date: 9-7-2019
"""
import colorlog
import logging
import os
import time
import traceback


class Logger(object):
    __instance = None

    def __init__(self):
        pass

    def __new__(
            cls,
            name: str = __name__,
            level: int = logging.DEBUG,
            path_output: str = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs"),
            **kwargs
    ):

        if not cls.__instance:
            cls.__instance = object.__new__(cls)
            cls.__configure_logger(name=name, level=level, path_output=path_output)

        return cls.__instance

    @classmethod
    def __configure_logger(cls, name: str, level: int, path_output: str):

        log_format = '%(asctime)s - %(levelname)s - %(message)s'

        # logging.basicConfig(level=level)
        logging.setLoggerClass(MyLogger)
        cls.__instance = logging.getLogger(name)
        cls.__instance.setLevel(level)

        # create a file handler
        if not os.path.exists(path_output):
            os.makedirs(path_output)

        fh = logging.FileHandler(os.path.join(path_output, time.strftime('%Y-%m-%d-%H-%M-%S.log')))
        fh.setLevel(level)

        ch = logging.StreamHandler()
        ch.setLevel(level)

        # create a logging format
        colored_formatter = colorlog.ColoredFormatter(
            "%(log_color)s" + log_format,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_purple'
            }
        )
        formatter = logging.Formatter(log_format)

        fh.setFormatter(formatter)
        ch.setFormatter(colored_formatter)

        # add the handlers to the logger
        cls.__instance.addHandler(fh)
        cls.__instance.addHandler(ch)

    def info(self, param):
        pass

    def critical(self, param):
        pass

    def error(self, param):
        pass

    def warning(self, param):
        pass

    def debug(self, param):
        pass


class MyLogger(logging.Logger):

    def __init__(self, name: str = __name__, level: int = logging.DEBUG):
        super().__init__(name=name, level=level)

    @staticmethod
    def get_indentation_level() -> int:
        traces = traceback.extract_stack()

        # strip builtin methods
        traces = [t for t in traces if "anaconda3" not in str(t).lower() and "pycharm" not in str(t).lower()]

        # strip logger attributes
        traces = [t for t in traces if not any([str(t).strip("<>").endswith(v) for v in dir(MyLogger)])]

        # strip profiler
        traces = [t for t in traces if "func_wrapper" not in str(t)]
        # return len(traces)
        return 0

    def indent_message(self, msg: str) -> str:
        indentation_level = self.get_indentation_level()
        return f"{'.' * indentation_level} {msg}" if indentation_level != 0 else msg

    def info(self, msg: str = "", *args, **kwargs):
        msg = self.indent_message(msg)
        return super(MyLogger, self).info(msg, *args, **kwargs)

    def critical(self, msg: str = "", *args, **kwargs):
        msg = self.indent_message(msg)
        return super(MyLogger, self).critical(msg, *args, **kwargs)

    def error(self, msg: str = "", *args, **kwargs):
        msg = self.indent_message(msg)
        return super(MyLogger, self).error(msg, *args, **kwargs)

    def warning(self, msg: str = "", *args, **kwargs):
        msg = self.indent_message(msg)
        return super(MyLogger, self).warning(msg, *args, **kwargs)

    def debug(self, msg: str = "", *args, **kwargs):
        msg = self.indent_message(msg)
        return super(MyLogger, self).debug(msg, *args, **kwargs)


def foo():
    Logger().info("Nested function foo")


if __name__ == "__main__":
    Logger().info("Boe")
    foo()
