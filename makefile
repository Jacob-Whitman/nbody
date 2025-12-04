CC = gcc
NVCC = nvcc

FLAGS = -DDEBUG
CFLAGS = $(FLAGS)
NVCCFLAGS = $(FLAGS)
LIBS = -lm
ALWAYS_REBUILD = makefile

nbody: nbody.o compute.o
	$(NVCC) $(NVCCFLAGS) -o nbody nbody.o compute.o $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(NVCCFLAGS) -c $<

compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -f *.o nbody