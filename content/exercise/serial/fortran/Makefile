# Copyright (c) 2019 CSC Training
# Copyright (c) 2021 ENCCS
ifeq ($(COMP),)
COMP=nv
endif

HAVE_PNG=0
ifeq ($(HAVE_PNG),1)
PNG_LIBS=-lpng -lz -lc -lrt
PNG_LDFLAGS=-L/usr/lib -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib 
endif


COMMONDIR=../../common

ifeq ($(COMP),nv)
FC=nvfortran
CC=nvc
FCFLAGS=-O3 -mp=multicore -Minfo=mp
CCFLAGS=-O3 -I$(COMMONDIR)
LDFLAGS=
LIBS= 
endif

ifeq ($(COMP),gnu)
FC=gfortran
CC=gcc
FCFLAGS=-O3 -Wall -fopenmp
CCFLAGS=-O3 -Wall -I$(COMMONDIR)
LDFLAGS=-fopenmp
LIBS=
endif


EXE=heat_serial
OBJS=main.o heat_mod.o core.o setup.o utilities.o io.o pngwriter_mod.o
OBJS_PNG= $(COMMONDIR)/pngwriter.o

all: $(EXE)

$(COMMONDIR)/pngwriter.o: $(COMMONDIR)/pngwriter.c $(COMMONDIR)/pngwriter.h
heat_mod.o: heat_mod.F90
core.o: core.F90 heat_mod.o
utilities.o: utilities.F90 heat_mod.o
io.o: io.F90 heat_mod.o pngwriter_mod.o 
setup.o: setup.F90 heat_mod.o utilities.o io.o
pngwriter_mod.o: pngwriter_mod.F90 heat_mod.o
main.o: main.F90 heat_mod.o core.o io.o setup.o utilities.o

$(EXE): $(OBJS) $(OBJS_PNG)
	$(FC) $(FCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(PNG_LDFLAGS) $(LIBS) $(PNG_LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) -DHAVE_PNG=$(HAVE_PNG) -I/usr/include -I/usr/local/include $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.mod *.png *~  ../../common/*.o

