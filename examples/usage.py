# -*- coding: utf-8 -*-

from nelder_mead import NelderMead


def sphere(x):
    return sum([t**2 for t in x])


def main():
    func = sphere
    params = {
        "x1": (-512, 512),
        "x2": (-512, 512),
    }

    nm = NelderMead(func, params)
    nm.minimize(n_iter=30)


if __name__ == "__main__":
    main()
