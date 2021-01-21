import math
from unittest import TestCase

from priv_lib_util.tools import benchmarking


class Test_benchmarking(TestCase):
    def test_time_convertor_sec2hours_min_sec(self):
        seconds = [10, 60, 61, 3600, 7261., 7261.]
        res0 = [benchmarking.time_convertor_sec2hours_min_sec(second, time_format=0) for second in seconds]
        res1 = [benchmarking.time_convertor_sec2hours_min_sec(second, time_format=1) for second in seconds]
        res2 = [benchmarking.time_convertor_sec2hours_min_sec(second, time_format=2) for second in seconds]

        assert res0 == [(10, 0), (60, 0), (61, 0), (3600, 0), (7261, 0), (7261, 0)]
        assert res1 == [(10, 0, 0), (0, 1, 0), (1, 1, 0), (0, 60, 0), (1, 121, 0.), (1, 121, 0.)]
        assert res2 == [(10, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 1, 2, 0.), (1, 1, 2, 0.)]

        # alternative testing because of numerical inconsistencies.
        minisecond1 = 0.15
        minisecond2 = 0.7
        minires1 = benchmarking.time_convertor_sec2hours_min_sec(minisecond1, time_format=0)
        minires2 = benchmarking.time_convertor_sec2hours_min_sec(minisecond2, time_format=0)

        assert abs(minisecond1 - minires1[1]) + abs(minisecond2 - minires2[1]) < 0.001

    def test_time_time2text(self):
        s = 10
        m = 100
        h = 5
        assert benchmarking.time_time2text(s, m, h) == ('10 seconds ', '100 minutes ', '5 hours ')

        s = 1
        m = 1
        h = 1
        assert benchmarking.time_time2text(s, m, h) == ('1 second ', '1 minute ', '1 hour ')

        s = 0
        m = 0
        h = 0
        frac_sec = 0.5
        assert benchmarking.time_time2text(s, m, h, frac_sec) == ('0.5 second ', '', '')

        s = 0.
        m = 0
        h = 0
        assert benchmarking.time_time2text(s, m, h) == ('0 second ', '', '')
