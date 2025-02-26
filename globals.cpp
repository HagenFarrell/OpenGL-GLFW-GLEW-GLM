// globals.cpp

#include "globals.h"

int WindowWidth = 640;
int WindowHeight = 480;

GLuint drawProgram = 0;
GLuint computeProgram = 0;

GLuint vao = 0;
GLuint texture_out = 0;
int workgroups[3] = { 1,1,1 };

GLint compute_work_group_size[3] = { 0, 0, 0 };
GLuint paletteBuffer = 0;