ODIR		=	./bin
CPP		=	g++
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I. -I./src -I./zi -I/usr/local/boost/1.55.0/include
LIB_FLAGS	=	-L/usr/local/boost/1.55.0/lib64
OPT_FLAGS	=	-DNDEBUG -O2
OTH_FLAGS	=	-Wall -Wextra -Wno-unused-variable
LIBS		=	-lfftw3 -lpthread -lrt -lfftw3_threads
BOOST_LIBS	=	-lboost_program_options -lboost_regex -lboost_filesystem -lboost_system

znn: src/main.cpp
	$(CPP) -o $(ODIR)/znn src/main.cpp $(CPP_FLAGS) $(INC_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS) $(BOOST_LIBS) $(LIB_FLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*
