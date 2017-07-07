# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt


def plot2d_simplex(simplex, ind):
    fig_dir = "./"
    plt.cla()
    plt.xlim((-512, 512))
    plt.ylim((-512, 512))

    plt.plot([simplex[0].x[0], simplex[1].x[0]],
             [simplex[0].x[1], simplex[1].x[1]], color="#000000")
    plt.plot([simplex[1].x[0], simplex[2].x[0]],
             [simplex[1].x[1], simplex[2].x[1]], color="#000000")
    plt.plot([simplex[2].x[0], simplex[0].x[0]],
             [simplex[2].x[1], simplex[0].x[1]], color="#000000")
    plt.savefig(os.path.join(fig_dir, "{:03d}.png".format(ind)))
