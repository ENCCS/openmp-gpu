# Copyright (c) 2019 CSC Training
# Copyright (c) 2021 ENCCS
ifeq ($(COMP),)
COMP=gnu
endif

HAVE_PNG=0
ifeq ($(HAVE_PNG),1)
PNG_LIBS=-lpng -lz -lc -lrt
PNG_LDFLAGS=-L/usr/lib -L/usr/lib/x86_64-linux-gnu -L/usr/local/lib
endif

COMMONDIR=../common

ifeq ($(COMP),pgi)
CXX=pgCC
CC=pgcc
CCFLAGS=-g -O3 -fopenmp -I$(COMMONDIR)
LDFLAGS=
LIBS=
endif

ifeq ($(COMP),gnu)
CXX=g++
CC=gcc
CCFLAGS=-g -O3 -fopenmp -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=
endif

ifeq ($(COMP),nv)
CXX=nvc++
CC=nvc
CCFLAGS=-g -O3 -fopenmp -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=
endif

ifeq ($(COMP),intel)
CXX=icpx
CC=icx
CCFLAGS=-g -O3 -fopenmp -I$(COMMONDIR)
LDFLAGS=
LIBS=
endif

EXE=heat_serial
OBJS=main.o core.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o


all: $(EXE)


core.o: core.cpp heat.h
utilities.o: utilities.cpp heat.h
setup.o: setup.cpp heat.h
io.o: io.cpp heat.h
main.o: main.cpp heat.h

$(OBJS_PNG): C_COMPILER := $(CC)
$(OBJS_PNG): CCFLAGS += -DHAVE_PNG=$(HAVE_PNG) -I/usr/include -I/usr/local/include
$(OBJS): C_COMPILER := $(CXX)

$(EXE): $(OBJS) $(OBJS_PNG)
	$(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(PNG_LDFLAGS) $(LIBS) $(PNG_LIBS)

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	-/bin/rm -f $(EXE) a.out *.o *.png *~ ../common/*.o
