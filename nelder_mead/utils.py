# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np


def plot2d_simplex(simplex, ind):
    fig_dir = "./"
    plt.cla()
    n = 1000
    x1 = np.linspace(-256, 1024, n)
    x2 = np.linspace(-256, 1024, n)
    X, Y = np.meshgrid(x1, x2)
    Z = np.sqrt(X ** 2 + Y ** 2)
    plt.contour(X, Y, Z, levels=list(np.arange(0, 1200, 10)))
    plt.gca().set_aspect("equal")
    plt.xlim((-256, 768))
    plt.ylim((-256, 768))

    plt.plot([simplex[0].x[0], simplex[1].x[0]],
             [simplex[0].x[1], simplex[1].x[1]], color="#000000")
    plt.plot([simplex[1].x[0], simplex[2].x[0]],
             [simplex[1].x[1], simplex[2].x[1]], color="#000000")
    plt.plot([simplex[2].x[0], simplex[0].x[0]],
             [simplex[2].x[1], simplex[0].x[1]], color="#000000")
    plt.savefig(os.path.join(fig_dir, "{:03d}.png".format(ind)))
