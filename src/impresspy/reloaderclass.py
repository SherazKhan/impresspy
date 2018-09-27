import sys

class Reloader(object):
    def __init__(self, modulename):
        before = sys.modules.keys()
        __import__(modulename)
        after = sys.modules.keys()
        names = list(set(after) - set(before))
        self._toreload = [sys.modules[name] for name in names]
        for i in self._toreload:
            reload(i)

    def do_reload(self):
        for i in self._toreload:
            reload(i)