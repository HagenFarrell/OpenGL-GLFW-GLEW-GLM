// globals.h
#pragma once

#include <GL/glew.h>

extern int WindowWidth;
extern int WindowHeight;

extern GLuint drawProgram;
extern GLuint computeProgram;

extern GLuint vao;
extern GLuint texture_out;
extern int workgroups[3];

extern GLint compute_work_group_size[3];
extern GLuint paletteBuffer;