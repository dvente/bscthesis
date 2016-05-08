.DELETE_ON_ERROR:
.PHONY: compile all matlab octave clean test


CXXFLAGS := -std=gnu++11 -g -O3 -Wall -pipe -march=native -MMD

EXECS := test_squint dump
MEXS := lnevidence.mex
MEXAS := lnevidence.mexa64

DEPS := $(EXECS:=.d) $(MEXS:.mex=.d) $(MEXAS:.mexa64=2.d)

compile : $(EXECS)   # default target
octave : $(MEXS)
matlab : $(MEXAS)
python : squint.i squint.h squint.cpp
all : compile matlab octave

# automatically track "include" dependencies (they are output by gcc -MMD)
-include $(DEPS)


# Default rule puts all dependencies as arguments. We put only the main cpp file
$(EXECS) : % : %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Octave
$(MEXS) : %.mex : %.cpp
	CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" mkoctfile --mex $<
	sed 's/$*.o/$@/g' -i $*.d

# Matlab
$(MEXAS) : %.mexa64 : %.cpp
	$(shell matlab -e | grep MATLAB= | cut -d = -f 2-)/bin/mex -cxx CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS) -fPIC -MF $*2.d" $<

dump.dat : dump
	./$< > $@

clean:
	rm -rf $(EXECS) $(MEXS) $(MEXAS) $(DEPS) $(OBJECTS)

python:
	swig -c++ -python squint.i
	python setup.py build_ext --inplace

test : test_squint
	./test_squint


