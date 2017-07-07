# -*- coding: utf-8 -*-

import numpy as np


class Point(object):

    def __init__(self, dim):
        self.x = np.zeros((dim))
        self.f = 0

    def __str__(self):
        return "Params: {}, ObjValue: {}".format(", ".join(["{:>10.5f}".format(x) for x in self.x]), self.f)

    def __eq__(self, rhs):
        return self.f == rhs.f

    def __lt__(self, rhs):
        return self.f < rhs.f

    def __le__(self, rhs):
        return self.f <= rhs.f

    def __gt__(self, rhs):
        return self.f > rhs.f

    def __ge__(self, rhs):
        return self.f >= rhs.f
