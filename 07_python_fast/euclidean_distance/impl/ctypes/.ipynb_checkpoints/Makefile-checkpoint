all: euclidean_distance.so

euclidean_distance.so:
	gcc -fPIC -O3 -std=gnu17 -c euclidean_distance.c -o euclidean_distance.o
	gcc -shared -o euclidean_distance.so euclidean_distance.o
	rm euclidean_distance.o || true

clean:
	rm euclidean_distance.so euclidean_distance.o || true