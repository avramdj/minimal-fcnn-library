compile: main.cpp libfcnn.o dataset.txt
	g++ -I ../../ main.cpp libfcnn.o -std=c++11 -o a.out
	rm -rf libfcnn.o dataset-gen
dataset.txt: dataset-gen.cpp
	g++ dataset-gen.cpp -o dataset-gen
	./dataset-gen

libfcnn.o:
	g++ -I ../../ ../../src/libfcnn.cpp -c -o libfcnn.o

.PHONY: clean

clean:
	rm -rf libfcnn.o a.out dataset.txt dataset-gen
