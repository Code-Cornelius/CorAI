import functools
import time
from math import floor
from time import time


#todo add to benchmark average and variance of result.

def time_convertor_sec2hours_min_sec(seconds, time_format=0):
    """
    Semantics:
        Instead of converting the seconds in minutes outside, one can do it here.

    Args:
        seconds: runtime
        time_format:  0 is in seconds (no change), 1 is in min, 2 is in hour.

    Returns:
        converted time. 2/3/4-tuple s,m,h, seconds_frac

    """
    seconds_int = floor(seconds)
    seconds_frac = seconds - seconds_int
    if time_format == 0:
        return seconds_int, seconds_frac
    else:
        m, s = divmod(seconds_int, 60)
        if time_format == 1:
            return s, m, seconds_frac
        else:
            h, m = divmod(m, 60)
            return s, m, h, seconds_frac


def time_time2text(s, m, h, seconds_frac=0):
    """
    Semantics:
        Writes time as s,m,h into the format of text for printing for example.

    Args:
        s: seconds
        m: minutes
        h: hours
        seconds_frac: lower than a seconds

    Returns:
        Format is ('s+seconds_frac seconds ', 'm minutes ', 'h hours ').
        The plural is changed depending on how many units there are, and if a variable is 0, the string is empty.

    """
    if s == 0:
        ts = ""
    elif s == 1:
        ts = f"{s + seconds_frac} second "
    else:
        ts = f"{s:d} seconds "

    if m == 0:
        tm = ""
    elif m == 1:
        tm = f"{m:d} minute "
    else:
        tm = f"{m:d} minutes "

    if h == 0:
        th = ""
    elif h == 1:
        th = f"{h:d} hour "
    else:
        th = f"{h:d} hours "

    if h == s and s == m and m == 0:
        ts = "{} second ".format(seconds_frac)
    return ts, tm, th


def time_print_elapsed_time(start, end, title="no title"):
    """
        Function for printing the elapsed time.

    Args:
        start: beginning simulation's time.
        end: end simulation's time.
        title: function or bookmark to recognize where the time comes from.

    Returns:

    """
    seconds = end - start
    beg = " Program: " + title + ", took roughly:"
    print(100 * '~')
    s, m, h, seconds_frac = time_convertor_sec2hours_min_sec(seconds, time_format=2)
    ts, tm, th = time_time2text(s, m, h, seconds_frac)
    print(''.join([beg, th, tm, ts, 'to run.']))
    print(100 * '-')
    print(100 * '-')
    return


def benchmark(func, title="no title", *args, **kwargs):
    """
    Semantics:
        Helper for benchmarking a function and prints the timing with the name given as title.
        Extra parameters can be given.

    Args:
        func: function to benchmark
        title: title that is printed as:  " Program: " + title + ", took roughly:"
        *args: Additional arguments to pass as arguments to `func`.
        **kwargs: Additional keyword arguments to pass as keywords arguments to
            `func`.

    Returns:

    """
    start = time()
    func(*args, **kwargs)
    end = time()
    time_print_elapsed_time(start, end, title=title)


def wrap_benchmark(func):
    """
    Semantics:
        Print the runtime of the decorated function

    Args:
        func: Function to time.

    Returns: the time.

    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # time.perf_counter() is the most precise available clock.
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer
