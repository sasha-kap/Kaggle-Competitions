"""
Slightly modified Timer class code from https://realpython.com/python-timer/
to be used for printing elapsed execution time for functions.

Modifications to original code:
    - Printing to log includes function name and values of arguments passed
        to function

Usage:

    As decorator:
    -------------

    from timer import Timer

    @Timer(logger = logging.info)
    def func():
        pass

"""

from dataclasses import dataclass, field
from functools import wraps
import inspect
import time
from typing import Any, Callable, ClassVar, Dict, Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    text: str = "Function {} completed. Elapsed time: {:0.4f} seconds"
    func_name: str = None
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)

    def __call__(self, func):
        """Support using Timer as a decorator"""

        @wraps(func)
        def wrapper_timer(*args, **kwargs):
            self.logger(
                f"Running {func.__name__} function with {inspect.signature(func)} args/kwargs."
            )
            with self:
                self.func_name = func.__name__
                return func(*args, **kwargs)

        return wrapper_timer

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(self.func_name, elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
