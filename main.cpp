// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <json.hpp>

#include "setupShaderPrograms.h"

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// Include extern globals variables used in multiple files.
#include "globals.h"

// Using the Nlohmann JSON parser.
using json = nlohmann::json;

// Struct to hold sphere values, color added for ease of computing from JSON.
struct Sphere
{
    glm::vec4 center;
    glm::vec4 color;
};

// Struct to hold wall values, color added for ease of computing from JSON.
struct Wall
{
    glm::mat4 transform;
    glm::mat4 transformInv;
    glm::vec4 color;
};

// Vector to hold all walls.
std::vector<Wall> walls;

struct Molecule
{
    std::vector<Sphere> spheres;
    glm::vec3 min;
    glm::vec3 max;
};

glm::vec3 extendedExtent;

// We need a function that will setup the wall transform and inverse transform.
void setupWall(Molecule molecule)
{
    glm::vec3 center = (molecule.min + molecule.max) * 0.5f;
    glm::vec3 extent = (molecule.max - molecule.min) * 0.5f;

    // We need to create extended bounds that will be used to compute the walls.
    extendedExtent = extent * 2.0f;

    // Create a view-to-world transformation matrix that ensures correct orientation
    auto viewToWorld = glm::mat4(1.0f);

    // Flip the Y-axis to match OpenGL coordinate system. (Insane)
    //viewToWorld[1][1] = -1.0f;

    auto createWall = [&viewToWorld](const glm::mat4& transform, const glm::vec4& color)
    {
        Wall wall;
        wall.transform = viewToWorld * transform; // Apply coordinate system correction
        wall.transformInv = inverse(wall.transform);
        wall.color = color;
        return wall;
    };

    // Create the 5 walls.
    // Back wall
    {
        auto transform = glm::mat4(1.0f);
        transform = translate(transform, center + glm::vec3(0, 0, -extendedExtent.z / 2));
        transform = scale(transform, glm::vec3(extendedExtent.x / 2, extendedExtent.y / 2, 0.1f));
        walls.push_back(createWall(transform, glm::vec4(1, 1, 1, 1)));
    }
    // Bottom wall
    {
        auto transform = glm::mat4(1.0f);
        transform = translate(transform, center + glm::vec3(0, -extendedExtent.y / 2, 0));
        transform = scale(transform, glm::vec3(extendedExtent.x / 2, 0.1f, extendedExtent.z / 2));
        walls.push_back(createWall(transform, glm::vec4(0, 0.9, 0, 1)));
    }

    // Left wall
    {
        auto transform = glm::mat4(1.0f);
        transform = translate(transform, center + glm::vec3(extendedExtent.x / 2, 0, 0));
        transform = scale(transform, glm::vec3(0.1f, extendedExtent.y / 2, extendedExtent.z / 2));
        walls.push_back(createWall(transform, glm::vec4(0.9, 0, 0, 1)));
    }

    // Right wall
    {
        auto transform = glm::mat4(1.0f);
        transform = translate(transform, center + glm::vec3(-extendedExtent.x / 2, 0, 0));
        transform = scale(transform, glm::vec3(0.1f, extendedExtent.y / 2, extendedExtent.z / 2));
        walls.push_back(createWall(transform, glm::vec4(0, 0, 0.9, 1)));
    }

    // Top wall
    {
        auto transform = glm::mat4(1.0f);
        transform = translate(transform, center + glm::vec3(0, extendedExtent.y / 2, 0));
        transform = scale(transform, glm::vec3(extendedExtent.x / 2, 0.1f, extendedExtent.z / 2));
        walls.push_back(createWall(transform, glm::vec4(0, 0.9, 0, 1)));
    }

    // Debug that can be uncommented for the sake of seeing wall transforms.
    for (size_t i = 0; i < walls.size(); i++)
    {
        const auto& wall = walls[i];
        glm::vec4 centerPos = wall.transform * glm::vec4(0, 0, 0, 1);
        std::cout << "Wall " << i << " center position: (" <<
            centerPos.x << ", " << centerPos.y << ", " << centerPos.z << ")" << "\n";
    }


    // We need to upload the walls to the GPU.
    GLuint wallSSBO;
    glGenBuffers(1, &wallSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, wallSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, walls.size() * sizeof(Wall), walls.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Bind the SSBO to binding point 1
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, wallSSBO);
}

void checkGLError(const char* operation)
{
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR)
    {
        std::cout << "OpenGL error after " << operation << ": " << error << "\n";
    }
}

int main(void)
{
    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make macOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(1024, 768, "Project 5: Hagen Farrell", nullptr, nullptr);

    if (window == nullptr)
    {
        fprintf(
            stderr,
            "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Call setup functions from given setupShaderPrograms file.
    setupDrawProgram();
    setupComputeProgram();

    // Let's try and render the ethanol molecule first.
    std::ifstream file("molecules.json");
    json data = json::parse(file);

    Molecule molecule;
    json& moleculeData = data["Caffeine"];

    /*
    * This is where all the parsing from the JSON file is going to happen
    * using the library from online, we will extract quads, colors, then the
    * bounding box.
    */

    // Parse quadruples
    auto& quadruples = moleculeData["quadruples"];
    for (size_t i = 0; i < quadruples.size(); i += 4)
    {
        Sphere sphere;
        sphere.center = glm::vec4(
            quadruples[i].get<float>(),
            quadruples[i + 1].get<float>(),
            quadruples[i + 2].get<float>(),
            quadruples[i + 3].get<float>() // Stores the radius.
        );
        molecule.spheres.push_back(sphere);
    }

    // Parse the rest of the information like color.
    auto& colors = moleculeData["color"];
    for (size_t i = 0; i < molecule.spheres.size(); i++)
    {
        molecule.spheres[i].color = glm::vec4(
            colors[i * 3].get<float>() / 255.0f,
            colors[i * 3 + 1].get<float>() / 255.0f,
            colors[i * 3 + 2].get<float>() / 255.0f,
            1.0f
        );
    }

    // Parse the bounding box (AABB).
    auto& box = moleculeData["box"];
    molecule.min = glm::vec3(
        box["min"][0].get<float>(),
        box["min"][1].get<float>(),
        box["min"][2].get<float>()
    );
    molecule.max = glm::vec3(
        box["max"][0].get<float>(),
        box["max"][1].get<float>(),
        box["max"][2].get<float>()
    );


    // Setup and upload SSBO for spheres.
    GLuint sphereBuffer;
    glGenBuffers(1, &sphereBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sphereBuffer);

    glBufferData(GL_SHADER_STORAGE_BUFFER, molecule.spheres.size() * sizeof(Sphere), molecule.spheres.data(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // Unbind

    // Bind SSBO to binding point 0
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sphereBuffer);

    // Grab location of uniform variables.
    GLint locEye = glGetUniformLocation(computeProgram, "eye");
    GLint locAt = glGetUniformLocation(computeProgram, "at");
    GLint locUp = glGetUniformLocation(computeProgram, "up");
    GLint locFov = glGetUniformLocation(computeProgram, "fov");
    GLint locResolution = glGetUniformLocation(computeProgram, "resolution");

    // Assign values to send as uniforms.
    glm::vec3 upVal(0.f, 1.f, 0.f);
    float fovVal = 40.0f * (3.14159265f / 180.0f);
    glm::vec2 resolutionVal(WindowWidth, WindowHeight);

    // Get and verify uniform locations
    GLint aabbMinLoc = glGetUniformLocation(computeProgram, "uBoundingBox.min");
    GLint aabbMaxLoc = glGetUniformLocation(computeProgram, "uBoundingBox.max");

    // Ensure we can capture the escape key being pressed below.
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Setup camera.
    float rotationAngle = 0.0f;
    static float rotationSpeed = 0.1f;

    // Setup walls and send to GPU.
    setupWall(molecule);

    glm::vec3 boxCenter = (molecule.min + molecule.max) * 0.5f;
    float diagonal = length(molecule.max - molecule.min);

    // Area light diameter is 1/5 the "front" wall; we only have a back wall. so.
    float lightRadius = extendedExtent.x / 10.0f;

    // Center position of top wall.
    glm::vec3 centerTopWall = walls[1].transform * glm::vec4(0.0, 0.0, 0.0, 1.0);
    glm::vec3 lightPos = centerTopWall + glm::vec3(0.0f, -0.51f, 0.0f);
    glm::vec3 lightColor = glm::vec3(1.0f, 0.9f, 0.7f) * 2.0f;

    GLint lightPosLoc = glGetUniformLocation(computeProgram, "lightPos");
    GLint lightRadiusLoc = glGetUniformLocation(computeProgram, "lightRadius");
    GLint lightColorLoc = glGetUniformLocation(computeProgram, "lightColor");

    do
    {
        //rotationAngle += rotationSpeed;
        float radians = glm::radians(rotationAngle);
        glm::vec3 cameraPos = boxCenter + glm::vec3(
            diagonal * sin(radians),
            0.0f,
            diagonal * cos(radians)
        );
        // Run compute shader to fill texture
        glUseProgram(computeProgram);

        glUniform3fv(locEye, 1, &cameraPos[0]);
        glUniform3fv(locAt, 1, &boxCenter[0]);
        glUniform3fv(locUp, 1, value_ptr(upVal));
        glUniform1f(locFov, fovVal);
        glUniform2fv(locResolution, 1, value_ptr(resolutionVal));

        // Set the uniforms for the AABB.
        glUniform3f(aabbMinLoc, molecule.min.x, molecule.min.y, molecule.min.z);
        glUniform3f(aabbMaxLoc, molecule.max.x, molecule.max.y, molecule.max.z);

        // Send the light uniforms to the shader.
        glUniform3fv(lightPosLoc, 1, value_ptr(lightPos));
        glUniform1f(lightRadiusLoc, lightRadius);
        glUniform3fv(lightColorLoc, 1, value_ptr(lightColor));

        glDispatchCompute(workgroups[0], workgroups[1], workgroups[2]);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Render to screen using drawProgram
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(drawProgram);

        // After the program gets assinged and used, we need to bind the texture and use the sampler.
        glActiveTexture(GL_TEXTURE0); // Tells the program to sample from texture unit 0.
        glBindTexture(GL_TEXTURE_2D, texture_out); // This is the texture the compute shader wrote to.
        glUniform1i(glGetUniformLocation(drawProgram, "myTexture"), 0);
        // We then grab the location of the 2D sampler to use texture unit 0.
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    } // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}
