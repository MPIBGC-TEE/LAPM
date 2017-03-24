#!/usr/bin/env python3
# vim:set ff=unix expandtab ts=4 sw=4:
# this is a pure python version 
# run with pyhton3 run_tests.py in a venv

from concurrencytest import ConcurrentTestSuite, fork_for_tests
import unittest
import sys

from pathlib import Path

def main():
    p = Path(__file__).absolute().parents[2]
    sys.path.append(p.as_posix())

    print("\n###################### running tests ##########################\n")

    s = unittest.defaultTestLoader.discover('', pattern="Test*")
    concurrent_suite = ConcurrentTestSuite(s, fork_for_tests(16))
    r = unittest.TextTestRunner()

    res = r.run(concurrent_suite)
    if len(res.errors) + len(res.failures) > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
