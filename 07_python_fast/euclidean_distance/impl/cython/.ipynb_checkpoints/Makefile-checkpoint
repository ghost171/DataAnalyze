all: euclidean_distance.*.so

euclidean_distance.*.so:
	python setup.py build_ext --inplace

clean:
	rm -r build euclidean_distance.c euclidean_distance.*.so
