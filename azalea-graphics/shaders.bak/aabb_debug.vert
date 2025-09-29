#version 450

layout(push_constant) uniform PC {
    mat4 viewProj;
    vec4 gridOrigin_ws;
    int  radius;
    int  height;
} pc;

layout(set=0, binding=0, std430) readonly buffer Visibility {
    uint visible[];
};

// Unit cube geometry encoded in shader
const vec3 cubeVerts[8] = vec3[8](
    vec3(0.0, 0.0, 0.0),
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, 1.0, 1.0)
);

const int cubeIndices[24] = int[24](
    0,1, 1,2, 2,3, 3,0, // bottom
    4,5, 5,6, 6,7, 7,4, // top
    0,4, 1,5, 2,6, 3,7  // verticals
);

ivec3 chunkCoords(uint instance) {
    int side = pc.radius * 2 + 1;
    int layerSize = side * side;

    int y = int(instance) / layerSize;
    int rem = int(instance) % layerSize;
    int z = rem / side;
    int x = rem % side;

    return ivec3(x - pc.radius, y, z - pc.radius);
}

void main() {
    uint chunk = gl_InstanceIndex;

    if (visible[chunk] == 0u) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vColor = vec3(0.0);
        return;
    }

    ivec3 coord = chunkCoords(chunk);

    vec3 base = pc.gridOrigin_ws.xyz + vec3(coord) * float(16);
    vec3 bmin = base;
    vec3 bmax = base + vec3(float(16));

    int vidx = cubeIndices[gl_VertexIndex];
    vec3 unit = cubeVerts[vidx];

    vec3 world = mix(bmin, bmax, unit);

    gl_Position = pc.viewProj * vec4(world, 1.0);
}
