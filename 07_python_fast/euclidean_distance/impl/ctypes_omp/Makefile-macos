CC = /usr/local/opt/llvm/bin/clang
CPPFLAGS = -I/usr/local/opt/llvm/include -I/usr/local/opt/libomp/include
LDFLAGS = -L/usr/local/opt/llvm/lib -L/usr/local/opt/libomp/lib

all: euclidean_distance.so

euclidean_distance.so:
	$(CC) -fPIC -O3 -std=gnu17 $(CPPFLAGS) -fopenmp -c euclidean_distance.c -o euclidean_distance.o
	$(CC) -shared -o euclidean_distance.so euclidean_distance.o $(LDFLAGS) -lomp 
	rm euclidean_distance.o

clean:
	rm euclidean_distance.so euclidean_distance.o || true