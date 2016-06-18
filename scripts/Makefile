all: python

OBJECTS := build __pychache__ squint_wrap.cxx _squint*.so
DEPS := setup.py  squint.cpp  squint.h  squint.i

python: $(DEPS)
	swig -python -c++ squint.i
	python3 setup.py build_ext --inplace

clean:
	rm -rf $(OBJECTS)