#pragma once
#ifndef SETUP_SHADER_PROGRAMS_H
#define SETUP_SHADER_PROGRAMS_H

#include <GL/glew.h>


GLuint loadShader(const GLchar* shaderFilePath, const char* shaderType);
void setupDrawProgram();
void setupComputeProgram();
void setupBuffers();
void checkShaderCompilation(GLuint shader, const char* shaderType);
void checkShaderProgram(GLuint program);

#endif