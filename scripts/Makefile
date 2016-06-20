all: squintboost

OBJECTS := build __pychache__ squint_wrap.cxx _squint*.so
DEPS := setup.py  squint.cpp  squint.h  squint.i
DATA := ../data/

squintboost: $(DEPS)
	swig -python -c++ squint.i
	python3 setup.py build_ext --inplace
	python3 generate.py 5000 ../data/SQTrain.dat
	python3 generate.py 5000 ../data/SQTest.dat

adaboost:
	python3 generate.py 5000 ../data/AdaTrain.dat
	python3 generate.py 5000 ../data/AdaTest.dat

nhboost:
	python3 generate.py 5000 ../data/NHTrain.dat
	python3 generate.py 5000 ../data/NHTest.dat

a9a:
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a -P ../data/
	wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t -P ../data/
	-cat ../data/a9a | grep -v 123 > ../data/a9a #removes one line to fix compatibility wiht sparse matrices


clean:
	rm -rf $(OBJECTS)
	rm -rf $(DATA)