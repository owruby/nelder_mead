# The Nelder-Mead method

The easy implementation of the Nelder-Mead method.

## Description

The Nelder-Mead is the one of derivative-free optimization method.
This method is called simplex method or ameba method.

![result](https://github.com/owruby/nelder_mead/blob/master/figures/anim.gif)

## Installation

```
pip install git+https://github.com/owruby/nelder_mead.git
```

### Dependencies

- numpy

## Usage

``` python
from nelder_mead import NelderMead


def sphere(x):
    return sum([t**2 for t in x])


def main():
    func = sphere
    params = {
        "x1": ["real", (-512, 512)],
        "x2": ["real", (-512, 512)],
    }

    nm = NelderMead(func, params)
    nm.minimize(n_iter=30)


if __name__ == "__main__":
    main()
```

## References

- https://academic.oup.com/comjnl/article-abstract/7/4/308/354237/A-Simplex-Method-for-Function-Minimization
- http://epubs.siam.org/doi/book/10.1137/1.9780898718768
