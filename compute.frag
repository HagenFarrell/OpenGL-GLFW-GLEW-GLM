#version 430 core
in vec2 texCoord;

out vec4 fragColor;

uniform sampler2D myTexture;

void main() { 
	fragColor = vec4(texture(myTexture, vec2(texCoord.x, 1.-texCoord.y))); // Render image upside down
}
