# OpenGL Ray Tracer

A GPU-accelerated ray tracer implemented in C++ and OpenGL for a master's level computer graphics course.

## Implementation Features

- **GPU Acceleration**: Utilizes OpenGL compute shaders for real-time ray tracing
- **Primitive Support**: Renders spheres and box-shaped walls with proper intersections
- **Materials & Lighting**: Basic diffuse and specular shading with shadow calculations
- **Area Light**: Implements area light sampling for soft shadows using concentric mapping
- **Molecule Visualization**: Loads molecule data from JSON files
- **Shadow Physics**: Realistic penumbra calculations based on occluder distance
- **Post-Processing**: Basic tone mapping and gamma correction

## Technical Details

- Core ray tracing implemented entirely in GLSL compute shaders
- Custom data structures for rays, spheres, walls, and intersection results
- Optimized bounding box acceleration structure
- Proper ray-primitive intersection algorithms
- Shader Storage Buffer Objects (SSBOs) for efficient data transfer to GPU

## Running the Project

Requires OpenGL 3.3+ with GLFW and GLEW libraries properly configured.

## Future Improvements

- Additional materials and lighting models
- Improved anti-aliasing
- More optimization techniques

---

*Developed by Hagen Farrell for a Master's level computer graphics course at UCF - CAP 6721*
