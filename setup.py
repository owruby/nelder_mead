from setuptools import setup, find_packages

setup(
    name="Nelder-Mead",
    version="0.1",
    url="https://github.com/owruby/nelder_mead",
    packages=find_packages(),
    author="Masaki Yano",
    author_email="ruby.yano1995@gmail.com",
    description="The implementation of the Nelder-Mead method",
    install_requires=[
        "numpy"
    ]
)
