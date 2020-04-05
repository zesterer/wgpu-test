#version 450

layout(location = 0) in vec3 v_pos;
layout(location = 1) in vec2 v_tex;

layout(set = 1, binding = 0)
uniform Uniforms {
	mat4 view_proj;
};

layout(location = 0) out vec2 f_tex;

void main() {
	f_tex = v_tex;
	gl_Position = view_proj * vec4(v_pos, 1.0);
}
