"""
setup.py file for squint SWIG
"""

from distutils.core import setup, Extension


squint_module = Extension("_squint",
                           sources=["squint_wrap.cxx", "squint.cpp"])

setup(	name = "squint",
		version = "1.0",
		author      = "Daniel Vente",
		author_email = "danvente@gmail.com",
		description = """A python wrapper for the squint code.""",
		ext_modules = [squint_module],
		py_modules = ["squint"])