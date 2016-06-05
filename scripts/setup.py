#!/usr/bin/env python

"""
setup.py file for squint SWIG
"""

from distutils.core import setup, Extension


squint_module = Extension('_squint',
                           sources=['squint_wrap.cxx', 'squint.cpp'],
                           )

setup (name = 'squint',
       version = '0.1',
       author      = "Daniel Vente",
       description = """A python wrapper for squint code""",
       ext_modules = [squint_module],
       py_modules = ["squint"],
       )