#version 430 core

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Image binding: an rgba32f, 2D image at binding=0
layout(rgba32f, binding = 0) uniform image2D outputImage;

uniform vec3 eye, at, up;
uniform float fov;
uniform vec2 resolution;

// Light uniforms for sampling.
uniform vec3 lightPos;
uniform float lightRadius;
uniform vec3 lightColor;

//  Define the canonical box bounds as constants.
const vec3 canonicalBoxMin = vec3(-1.0, -1.0, -1.0);
const vec3 canonicalBoxMax = vec3(1.0, 1.0, 1.0);

// Defines a custom infinity, just a large number
#define MY_INFINITY 1e30
#define PI 3.14159265
#define BIAS 0.01

// Define a Ray structure
struct Ray {
  vec3 o;       // origin
  vec3 d;       // direction
  float tmin;
  float tmax;
};

// Wall struct definition.
struct Wall
{
  mat4 transform;
  mat4 transformInv;
  vec3 color;
};

// A struct for the sphere object.
struct Sphere
{
  vec4 center;
  vec4 color;
};

// Axis-aligned bounding box struct.
struct AABB
{
  vec3 min;
  vec3 max;
};

struct HitResult
{
  int sphereIndex;
  float t;
};

// A struct for the scene hit result.
struct SceneHitResult
{
  int objectType;
  int objectIndex;
  float t;
  vec3 normal;
  bool inShadow;
};

// A struct for the box object.
struct BoxHitResult
{
  bool hit;
  float t;
  vec3 normal;
};

// Since we are sending over one molecule for now,
uniform AABB uBoundingBox;

// Create a SSBO for the spheres.
layout(std430, binding = 0) buffer Spheres
{
  Sphere spheres[];
};

// Create a shader storage buffer for the walls.
layout(std430, binding = 1) buffer Walls
{
  Wall walls[5];
};

// A function that generates a ray through a given pixel:
Ray rayGenerate(vec2 pixel)
{
  Ray ray;

  // Orthonormal basis
  vec3 w = normalize(eye - at);
  vec3 u = normalize(cross(up, w));
  vec3 v = cross(w, u);

  // Compute camera's width and height (in world space) given the FOV
  float height = 2.0 * tan(fov * 0.5);
  float width = height * (resolution.x / resolution.y);

  // Ray origin = eye
  ray.o = eye;

  // The formula for the direction below is just an example
  // consistent with "Ray Tracing in a Weekend" style. 
  float py = (2.0 * pixel.y + 1.0) / resolution.y - 1.0;
  float px = (2.0 * pixel.x + 1.0) / resolution.x - 1.0;

  float halfH = 0.5 * height;
  float halfW = 0.5 * width;

  // -w aims forward from the camera, then we shift by u & v
  ray.d = normalize(-w + py * halfH * v - px * halfW * u);

  // Minkowski or small offset
  ray.tmin = 0.001;
  ray.tmax = MY_INFINITY;

  return ray;
}

// Function from "Ray tracing in one weekend", adapted to glsl
vec3 ray_color(Ray r)
{
  vec3 unit_direction = normalize(r.d);
  float t = 0.5 * (unit_direction.y + 1.0);

  // Interpolate between white and blue
  vec3 white = vec3(1.0, 1.0, 1.0);
  vec3 sky = vec3(0.5, 0.7, 1.0);

  return mix(white, sky, t);
}

// We need to transform the ray from world space to the local space of the wall.
Ray transformRay(Ray worldRay, Wall wall)
{
  Ray localRay;

  // Transform the ray origin and direction to local space.
  localRay.o = (wall.transformInv * vec4(worldRay.o, 1.0)).xyz;
  localRay.d = normalize(vec3(wall.transformInv * vec4(worldRay.d, 0.0)));

  // Use a small epsilon in order to offset numerical unstability.
  localRay.tmin = worldRay.tmin + 1e-6;
  localRay.tmax = worldRay.tmax;

  return localRay;
}

// Intersect a ray with the canonical box.
BoxHitResult rayCanonicalBoxIntersect(Ray ray)
{
  BoxHitResult result;
  result.hit = false;

  vec3 invDir = 1.0 / (ray.d + vec3(1e-20));
  vec3 t0 = (canonicalBoxMin - ray.o) * invDir;
  vec3 t1 = (canonicalBoxMax - ray.o) * invDir;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);

  float t_enter = max(max(tmin.x, tmin.y), tmin.z);
  float t_exit = min(min(tmax.x, tmax.y), tmax.z);


  if (t_exit >= t_enter && t_enter < ray.tmax && t_exit > ray.tmin)
  {
	result.hit = true;
	result.t = t_enter;

	// Adjusted normal calculation to use dominant axis.
	vec3 t_enter_vec = vec3(t_enter);
	vec3 hitAxis = step(t_enter_vec, vec3(t_enter));
	result.normal = -sign(ray.d) * hitAxis;
  }

  return result;
}


float raySphereIntersect(vec3 rayOrigin, vec3 rayDir, Sphere sphere)
{
  vec3 oc = rayOrigin - sphere.center.xyz;

  float radiusSq = sphere.center.w * sphere.center.w;
  if (dot(oc, oc) <= radiusSq) return 0.0;

  float b = 2.0 * dot(oc, rayDir);
  float c = dot(oc, oc) - radiusSq;
  float discriminant = b * b - 4.0 * c;

  if (discriminant < 0.0) return -1.0;

  float sqrtDisc = sqrt(discriminant);
  float t0 = (-b - sqrtDisc) * 0.5;
  float t1 = (-b + sqrtDisc) * 0.5;

  if (t0 > 0.0) return t0;
  else if (t1 > 0.0) return t1;
  return -1.0;
}

SceneHitResult findClosestIntersection(Ray worldRay)
{
  // Setup default return values.
  SceneHitResult result;
  result.t = MY_INFINITY;
  result.objectType = -1;
  result.objectIndex = -1;
  result.inShadow = false;

  vec3 extendedMin = uBoundingBox.min - (uBoundingBox.max - uBoundingBox.min);
  vec3 extendedMax = uBoundingBox.max + (uBoundingBox.max - uBoundingBox.min);

  vec3 invDir = 1.0 / (worldRay.d + vec3(1e-20));
  vec3 t0 = (extendedMin - worldRay.o) * invDir;
  vec3 t1 = (extendedMax - worldRay.o) * invDir;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);

  float t_enter = max(max(tmin.x, tmin.y), tmin.z);
  float t_exit = min(min(tmax.x, tmax.y), tmax.z);

  // If the ray doesn't intersect the bounding box, return the default result.
  if (t_exit < t_enter || t_exit < 0.0)
  {
	return result;
  }

  // Check walls 
  for (int i = 0; i < walls.length(); i++)
  {
	Ray localRay = transformRay(worldRay, walls[i]);
	BoxHitResult boxHit = rayCanonicalBoxIntersect(localRay);
 
	if (boxHit.hit)
	{
	  // Converting the local intersection back to the world space,
	  // this was the most necessary thing I have done in this program.
	  vec3 localHit = localRay.o + localRay.d * boxHit.t;
	  vec4 worldHit = walls[i].transform * vec4(localHit, 1.0);

	  float worldT = length(worldHit.xyz - worldRay.o);

	  if (worldT < result.t)
	  {
		result.objectType = 2;
		result.objectIndex = i;
		result.t = worldT;
		vec3 hitPoint = worldRay.o + worldRay.d * result.t;
		/*
		 Transform the normal back to world space.
		 We need to use the inverse transpose of the walls transformation matrix
		 to ensure the normal is correctly oriented in world space.
		 Idea from: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals.html
		*/
		result.normal = normalize(vec3(transpose(walls[i].transformInv) * vec4(boxHit.normal, 0.0)));
	  }
	}
  }

  // Check spheres
  for (int i = 0; i < spheres.length(); i++)
  {
	float t = raySphereIntersect(worldRay.o, worldRay.d, spheres[i]);
	if (t > 0.0 && t < result.t)
	{
	  result.objectType = 1;
	  result.objectIndex = i;
	  result.t = t;
	  vec3 hitPoint = worldRay.o + worldRay.d * t;
	  result.normal = normalize(hitPoint - spheres[i].center.xyz);
	}
  }
  return result;
}

// Convert using concentric coordinate sampling.
vec2 squareToDisk(vec2 square)
{
    // Mapping from [0,1] -> [-1,1]
    vec2 offset = 2.0 * square - 1;
    
    // Simple case.
    if (offset.x == 0 && offset.y == 0) return vec2(0.0, 0.0);
    
    float theta, r;
  
    if ((offset.x * offset.x) > (offset.y * offset.y))
      {
        // Top half.
        r = abs(offset.x);
        theta = (PI/4) * (offset.y/offset.x);
        
        // Adjustment for quadrants.
        if (offset.x < 0.0) theta += PI;
      }
    else
      {
        // Bottom half.
        r = abs(offset.y);
        theta = (PI/2) - (PI/4) * (offset.x/offset.y);
        
        // Adjustment for quadrants.
        if (offset.y < 0.0) theta += PI;
      }


    // Scale by light radius after calculations, not during.
    r *= lightRadius;
    
    // Map the coordinates back to cartesian.
    float x = cos(theta) * r;
    float y = sin(theta) * r;
    
    // Z-axis is zero so just omit it.
    return vec2(x, y);
}

vec4 calculateSampleColor(Ray ray, SceneHitResult hit, vec2 square, int totalSamples)
{
    // If no hit, return background color
    if (hit.objectType <= 0) {
        return vec4(ray_color(ray), 1.0f);
    }

    // Calculate hit point in world space
    vec3 hitPoint = ray.o + ray.d * hit.t;

    // Determine material color based on what was hit
    vec3 materialColor = (hit.objectType == 1) ?
    spheres[hit.objectIndex].color.rgb :
    walls[hit.objectIndex].color.rgb;

    // Start with ambient lighting (this ensures nothing is completely black)
    vec3 finalColor = materialColor * 0.2;

    // Calculate specific point on the area light (light sampling)
    vec2 diskOffset = squareToDisk(square);
    vec3 lightPoint = lightPos + vec3(diskOffset.x, diskOffset.y, 0.0);

    // Vector FROM light TO hitpoint (for consistency)
    vec3 lightToHit = hitPoint - lightPoint;
    float distanceToLight = length(lightToHit);

    // Normalized direction vectors
    vec3 L = normalize(-lightToHit); // Direction FROM hitpoint TO light
    vec3 N = hit.normal;             // Surface normal
    vec3 V = normalize(eye - hitPoint); // Direction to viewer

    float shadowFactor = 1.0;
    
    // Shadow ray setup - strictly from hit point toward light
    Ray shadowRay;
    shadowRay.o = hitPoint + N * BIAS; 
    shadowRay.d = -normalize(lightToHit); 
    shadowRay.tmin = 0.001;
    shadowRay.tmax = distanceToLight - 0.001; // Only test up to light distance

    // Shadow test
    SceneHitResult shadowHit = findClosestIntersection(shadowRay);
    bool inShadow = shadowHit.objectType > 0;

    if (shadowHit.objectType > 0 )
    {
        float occluderDistance = shadowHit.t;
        float lightDistance = distanceToLight;
        
        float ratio = occluderDistance / lightDistance;
        
        // Penumbra calculations - objects closer to recievcer get more shadowing.
        // Objects closer to the light create softer shadowing.
        shadowFactor = smoothstep(0.0, 0.2, ratio * (1.0 - ratio));
        
        if (ratio < 0.1) shadowFactor = 0.0;
    }
    
    if (shadowFactor > 0.0) {
        // Calculate diffuse lighting
        float diffuseFactor = max(dot(N, L), 0.0);

        // Physical light attenuation with inverse square falloff
        float attenuation = 3.0 / (1.0 + 0.25 * distanceToLight * distanceToLight);

        // Apply shadow factor to diffuse component
        vec3 diffuseColor = materialColor * diffuseFactor * shadowFactor;

        // Calculate specular with Blinn-Phong model for better highlights
        vec3 H = normalize(L + V);  // Halfway vector
        float specFactor = pow(max(dot(N, H), 0.0), 64.0) * shadowFactor;
        vec3 specularColor = vec3(0.5) * specFactor;

        // Combine lighting components with physically-based attenuation
        finalColor += (diffuseColor + specularColor) * attenuation;
    }

    return vec4(finalColor, 1.0);
}

// The main() function runs once per work-item (pixel).
void main()
{
  // Get the pixel coordinates from the global invocation ID
  ivec2 pixelCoords = ivec2(gl_GlobalInvocationID.xy);
  if (pixelCoords.x >= int(resolution.x) || pixelCoords.y >= int(resolution.y)) return;

  // Carry the color information throughout the sampling.
    vec4 accumulatedColor = vec4(0.0, 0.0, 0.0, 0.0);

  int sampleSize = 16;
  int totalSamples = sampleSize * sampleSize; 

    for (int sampleIndex = 0; sampleIndex < totalSamples; sampleIndex++)
    {
        // Rows and columns for each subpixel.
        int x = sampleIndex / sampleSize;
        int y = sampleIndex % sampleSize;

        // Calculate offset for subpixel.
        vec2 offset = vec2(
            (float(x) +0.5f) / float(sampleSize),
            (float(y) +0.5f) / float(sampleSize));

        // Generate ray, and then test for intersections.   
        Ray r = rayGenerate(vec2(pixelCoords));
        SceneHitResult hit = findClosestIntersection(r);   
        
        vec4 sampleColor = calculateSampleColor(r, hit, offset, totalSamples);
        accumulatedColor += sampleColor;
    }     

    vec4 finalColor = accumulatedColor / float(totalSamples);
    
    // Testing out tone mapping and gamma correction for fun. (based off my own research!)
    finalColor.rgb = finalColor.rgb / (finalColor.rgb + vec3(1.0));
    finalColor.rgb = pow(finalColor.rgb, vec3(1.0/1.4)); // Simple gamma correction

  // Write the color out to the image, with alpha=1
  imageStore(outputImage, pixelCoords, finalColor);
}