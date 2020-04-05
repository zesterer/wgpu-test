#version 450

layout(location = 0) in vec2 f_tex;

layout(location = 0) out vec4 f_col;

//layout(set = 0, binding = 0) uniform texture2D t_tex;
//layout(set = 0, binding = 1) uniform sampler t_sampler;

void main() {
	f_col = vec4(f_tex, 0.0, 1.0);
}
