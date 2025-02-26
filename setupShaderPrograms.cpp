#include <GL/glew.h>
#include <GLFW/glfw3.h> 
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "globals.h"

// ---- Forward Declarations ----
GLuint loadShader(const GLchar* shaderFilePath, const char* shaderType);
void checkShaderCompilation(GLuint shader, const char* shaderType);
void checkShaderProgram(GLuint program);

GLuint setVertices();
GLuint setImageStore();
void setupBuffers();

void setupDrawProgram();
void setupComputeProgram();

GLuint setVertices() {
  float vertices[] = {
	// 3D position     // tex coords
	-1.f, -1.f, 0.f,   0.f, 0.f,
	-1.f,  1.f, 0.f,   0.f, 1.f,
	 1.f, -1.f, 0.f,   1.f, 0.f,
	 1.f,  1.f, 0.f,   1.f, 1.f
  };
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  // Position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // TexCoord attribute
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // We'll delete the VBO at program end or keep it around if needed
  return vao;
}

GLuint setImageStore() {
  GLuint texID;
  glGenTextures(1, &texID);
  glBindTexture(GL_TEXTURE_2D, texID);

  // Simple nearest sampling
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Allocate for WindowWidth x WindowHeight, with RGBA32F
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WindowWidth, WindowHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

  // Bind as image for compute
  glBindImageTexture(0, texID, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

  glBindTexture(GL_TEXTURE_2D, 0);
  return texID;
}

void setupBuffers() {
  // If you need a palette buffer or some SSBO, create it here
  glGenBuffers(1, &paletteBuffer);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, paletteBuffer);
  // For demonstration, 16 bytes (like 4 floats)
  glBufferData(GL_SHADER_STORAGE_BUFFER, 16, nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void setupDrawProgram()
{
  std::cout << "Setting up the draw program...\n";
  vao = setVertices();
  texture_out = setImageStore();

  // Make vertex/fragment shaders
  // (Make sure the file paths are correct on disk)
  GLuint vertShader = loadShader("compute.vert", "VERTEX");
  GLuint fragShader = loadShader("compute.frag", "FRAGMENT");

  drawProgram = glCreateProgram();
  glAttachShader(drawProgram, vertShader);
  glAttachShader(drawProgram, fragShader);
  glLinkProgram(drawProgram);
  checkShaderProgram(drawProgram);

  glDeleteShader(vertShader);
  glDeleteShader(fragShader);

  std::cout << "Draw program set up complete.\n";
}

void setupComputeProgram()
{
  std::cout << "Setting up the compute program...\n";
  computeProgram = glCreateProgram();
  GLuint cshader = loadShader("computeshader.glsl", "COMPUTE");

  glAttachShader(computeProgram, cshader);
  glLinkProgram(computeProgram);
  checkShaderProgram(computeProgram);
  glDeleteShader(cshader);

  // Query local group size from the compiled shader
  glGetProgramiv(computeProgram, GL_COMPUTE_WORK_GROUP_SIZE, compute_work_group_size);
  std::cout << "Local Work Group size: "
	<< compute_work_group_size[0] << ", "
	<< compute_work_group_size[1] << ", "
	<< compute_work_group_size[2] << "\n";

  // Calculate how many groups to dispatch
  workgroups[0] = (WindowWidth + compute_work_group_size[0] - 1) / compute_work_group_size[0];
  workgroups[1] = (WindowHeight + compute_work_group_size[1] - 1) / compute_work_group_size[1];
  workgroups[2] = 1;

  std::cout << "Compute Work Groups: "
	<< workgroups[0] << ", "
	<< workgroups[1] << ", "
	<< workgroups[2] << "\n";

  // Set up buffers, if needed
  setupBuffers();
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, paletteBuffer);

  std::cout << "Compute program set up complete.\n";
}

// Helper functions for loading/compiling shaders
GLuint loadShader(const GLchar* shaderFilePath, const char* shaderType) {
  GLuint shader = 0;
  if (std::string(shaderType) == "VERTEX")   shader = glCreateShader(GL_VERTEX_SHADER);
  if (std::string(shaderType) == "FRAGMENT") shader = glCreateShader(GL_FRAGMENT_SHADER);
  if (std::string(shaderType) == "COMPUTE")  shader = glCreateShader(GL_COMPUTE_SHADER);

  // Read file
  std::ifstream inFile(shaderFilePath);
  if (!inFile.good()) {
	std::cerr << "Cannot open shader file: " << shaderFilePath << std::endl;
	return 0;
  }
  std::stringstream code;
  code << inFile.rdbuf();
  inFile.close();
  std::string codeStr = code.str();

  // Compile
  const char* codeCStr = codeStr.c_str();
  glShaderSource(shader, 1, &codeCStr, nullptr);
  glCompileShader(shader);
  checkShaderCompilation(shader, shaderType);
  return shader;
}

void checkShaderCompilation(GLuint shader, const char* shaderType) {
  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
	GLchar infoLog[512];
	glGetShaderInfoLog(shader, 512, nullptr, infoLog);
	std::cerr << "ERROR: " << shaderType << " shader compilation failed:\n" << infoLog << std::endl;
  }
}

void checkShaderProgram(GLuint program) {
  GLint success;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
	GLchar infoLog[512];
	glGetProgramInfoLog(program, 512, nullptr, infoLog);
	std::cerr << "ERROR: Program linking failed:\n" << infoLog << std::endl;
  }
}