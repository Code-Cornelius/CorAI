import math
from unittest import TestCase

from priv_lib_util.tools import benchmarking


class Test_benchmarking(TestCase):
    def test_time_convertor_sec2hours_min_sec(self):
        seconds = [10, 60, 61, 3600, 7261., 7261.]
        res0, trueres0 = [benchmarking.time_convertor_sec2hours_min_sec(second, time_format=0) for second in seconds], \
                         [(10, 0), (60, 0), (61, 0), (3600, 0), (7261, 0), (7261, 0)]
        res1, trueres1 = [benchmarking.time_convertor_sec2hours_min_sec(second, time_format=1) for second in seconds], \
                         [(10, 0, 0), (0, 1, 0), (1, 1, 0), (0, 60, 0), (1, 121, 0.), (1, 121, 0.)]
        res2, trueres2 = [benchmarking.time_convertor_sec2hours_min_sec(second, time_format=2) for second in seconds], \
                         [(10, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 1, 2, 0.), (1, 1, 2, 0.)]
        res = [res0, res1, res2]
        trueres = [trueres0, trueres1, trueres2]

        for i in range(3):
            with self.subTest(i=i):
                assert res[i] == trueres[i]

        # alternative testing because of numerical inconsistencies.
        minisecond1 = 0.15
        minisecond2 = 0.7
        minires1 = benchmarking.time_convertor_sec2hours_min_sec(minisecond1, time_format=0)
        minires2 = benchmarking.time_convertor_sec2hours_min_sec(minisecond2, time_format=0)

        assert abs(minisecond1 - minires1[1]) + abs(minisecond2 - minires2[1]) < 0.001

    def test_time_time2text(self):
        frac_sec = [0, 0, 0.5, 0]
        s = [10, 1, 0, 0]
        m = [100, 1, 0, 0]
        h = [5, 1, 0, 0]
        res = [('10 seconds ', '100 minutes ', '5 hours '),
               ('1 second ', '1 minute ', '1 hour '),
               ('0.5 second ', '', ''),
               ('0 second ', '', '')]
        for i in range(len(s)):
            with self.subTest(i=i):
                assert benchmarking.time_time2text(s[i], m[i], h[i], frac_sec[i]) == res[i]
