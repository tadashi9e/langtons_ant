CXXFLAGS=-g -Wall `pkg-config --cflags OpenCL glut glu gl`
LDFLAGS=`pkg-config --libs OpenCL glut glu gl`

all: langtons_ant

langtons_ant: langtons_ant.cpp
	g++ $(CXXFLAGS) langtons_ant.cpp -o langtons_ant $(LDFLAGS)

clean:
	rm -rf langtons_ant
