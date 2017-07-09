# -*- coding: utf-8 -*-

import numpy as np

from nelder_mead.point import Point


class NelderMead(object):

    def __init__(self, func, params, *args, **kwargs):
        """ the Nelder-Mead method

        :param func: objective function object
        :param params: dict, tuning parameter, name: (min, max)

        ---
        NelderMead(
             lambda x: x[0]**2 + 5 * x[1],
             {
                 "x1": (1, 5),
                 "x2": (0, 3),
             }
        )

        """
        self.func = func
        self.dim = len(params)
        self.n_eval = 0
        self.names = []
        self.p_types = []
        self.p_min = []
        self.p_max = []
        self.simplex = []
        self.initialized = False

        self._parse_minmax(params)

    def initialize(self, init_params):
        """ Inialize first simplex point

        :param init_params(list):

        """
        assert len(init_params) == (self.dim + 1), "Invalid the length of init_params"
        for param in init_params:
            p = Point(self.dim)
            p.x = np.array(param, dtype=np.float32)
            self.simplex.append(p)
        self.initlized = True

    def maximize(self, n_iter=20, delta_r=1, delta_e=2, delta_ic=-0.5, delta_oc=0.5, gamma_s=0.5):
        """ Maximize the objective function.

        :param n_iter: the number of iterations for the nelder_mead method
        :param delta_r: the parameter of reflect
        :param delta_e: the parameter of expand
        :param delta_ic: the parameter of inside contraction
        :param delta_oc: the parameter of outside contraction
        :param gamma_s: the parameter of shrink

        """
        self._coef = -1
        variables = locals()
        for k, v in variables.items():
            setattr(self, k, v)
        self._opt(n_iter)

    def minimize(self, n_iter=20, delta_r=1, delta_e=2, delta_ic=-0.5, delta_oc=0.5, gamma_s=0.5):
        """ Minimize the objective function.

        :param n_iter: the number of iterations for the nelder_mead method
        :param delta_r: the parameter of reflect
        :param delta_e: the parameter of expand
        :param delta_ic: the parameter of inside contraction
        :param delta_oc: the parameter of outside contraction
        :param gamma_s: the parameter of shrink

        """
        self._coef = 1
        variables = locals()
        for k, v in variables.items():
            setattr(self, k, v)
        self._opt(n_iter)

    def func_impl(self, x):
        objval, invalid = None, False
        for i, t in enumerate(x):
            if t < self.p_min[i] or t > self.p_max[i]:
                objval = float("inf")
                invalid = True
        if not invalid:
            x = [int(np.round(x_t)) if p_t is "integer" else x_t for p_t, x_t in zip(self.p_types, x)]
            objval = self._coef * self.func(x)

        print("{:5d} | {} | {:>15.5f}".format(
            self.n_eval,
            " | ".join(["{:>15.5f}".format(t) for t in x]),
            self._coef * objval
        ))

        self.n_eval += 1
        return objval

    def _opt(self, n_iter):
        # Print Header
        print("{:>5} | {} | {:>15}".format(
            "Eval",
            " | ".join(["{:>15}".format(name) for name in self.names]),
            "ObjVal"
        ))
        print("-" * (20 + self.dim * 20))

        if not self.initialized:
            self._initialize()
        for p in self.simplex:
            p.f = self.func_impl(p.x)

        for i in range(n_iter):
            self.simplex = sorted(self.simplex, key=lambda p: p.f)

            # centroid
            p_c = self._centroid()
            # reflect
            p_r = self._reflect(p_c)

            if p_r < self.simplex[0]:
                p_e = self._expand(p_c)
                if p_e < p_r:
                    self.simplex[-1] = p_e
                else:
                    self.simplex[-1] = p_r
                continue
            elif p_r > self.simplex[-2]:
                if p_r <= self.simplex[-1]:
                    p_cont = self._outside(p_c)
                    if p_cont < p_r:
                        self.simplex[-1] = p_cont
                        continue
                    self.simplex[-1] = p_r
                elif p_r > self.simplex[-1]:
                    p_cont = self._inside(p_c)
                    if p_cont < self.simplex[-1]:
                        self.simplex[-1] = p_cont
                        continue

                # Shirnk
                for j in range(len(self.simplex) - 1):
                    p = Point(self.dim)
                    p.x = self.simplex[0].x + self.gamma_s * \
                        (self.simplex[j + 1].x - self.simplex[0].x)
                    p.f = self.func_impl(p.x)
                    self.simplex[j + 1] = p
            else:
                self.simplex[-1] = p_r

        self.simplex = sorted(self.simplex, key=lambda p: p.f)
        print("\nBest Point: {}".format(self.simplex[0]))

    def _centroid(self):
        p_c = Point(self.dim)
        x_sum = []
        for p in self.simplex[:-1]:
            x_sum.append(p.x)
        p_c.x = np.mean(x_sum, axis=0)
        return p_c

    def _reflect(self, p_c):
        return self._generate_point(p_c, self.delta_r)

    def _expand(self, p_c):
        return self._generate_point(p_c, self.delta_e)

    def _inside(self, p_c):
        return self._generate_point(p_c, self.delta_ic)

    def _outside(self, p_c):
        return self._generate_point(p_c, self.delta_oc)

    def _generate_point(self, p_c, x_coef):
        p = Point(self.dim)
        p.x = p_c.x + x_coef * (p_c.x - self.simplex[-1].x)
        p.f = self.func_impl(p.x)
        return p

    def _parse_minmax(self, params):
        types = ["real", "integer"]
        for name, values in params.items():
            assert values[0] in types, "Invalid param type, Please check it."

            self.names.append(name)
            self.p_types.append(values[0])
            self.p_min.append(values[1][0])
            self.p_max.append(values[1][1])

    def _initialize(self):
        for i in range(self.dim + 1):
            p = Point(self.dim)
            init_val = [(m2 - m1) * np.random.random() + m1 for m1,
                        m2 in zip(self.p_min, self.p_max)]
            p.x = np.array(init_val, dtype=np.float32)
            self.simplex.append(p)
