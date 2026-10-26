
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "GG-Fall_shaders.h"
#include "xsdk/gpu.h"
#include "xsdk/kernel.h"
#include "xsdk/log.h"
#include "xsdk/present.h"
#include "xsdk/shader.h"
#include "xsdk/splash.h"
#include "xsdk/texture.h"

#define BUFFERS 2
#define SWAP_INTERVAL 1
#define BEGIN_TIMEOUT_MS 5000
#define CLEAR_ARGB 0xFF87CEEB
#define VERTEX_FLOATS 8
#define STRIDE_BYTES (VERTEX_FLOATS * 4)
#define VERTICES_BYTES (24 * STRIDE_BYTES)
#define INDICES_BYTES (36 * 2)
#define STICK_DEAD_ZONE 8000
#define SPEED 0.035f
#define JUMP_SPEED 0.12f
#define GRAVITY 0.004f
#define CAMERA_FOLLOW 0.60f

typedef struct { float m[16]; } Matrix;

typedef struct {
    XsdkShader vertex, pixel;
    XsdkTexture texture;
    XsdkSamplerState sampler;
    XsdkDepthStencilState depth;
    volatile uint32_t *vertices, *indices;
} Game;

static const XsdkVertexElement layout[] = {
    { .usage = XSDK_USAGE_POSITION, .format = XSDK_VERTEX_FLOAT3,
      .offset_bytes = 0 },
    { .usage = XSDK_USAGE_COLOR, .format = XSDK_VERTEX_FLOAT3,
      .offset_bytes = 12 },
    { .usage = XSDK_USAGE_TEXCOORD, .format = XSDK_VERTEX_FLOAT2,
      .offset_bytes = 24 },
};

static Matrix multiply(const Matrix *a, const Matrix *b)
{
    Matrix r;
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++)
                sum += a->m[k * 4 + row] * b->m[c * 4 + k];
            r.m[c * 4 + row] = sum;
        }
    }
    return r;
}

static Matrix model_matrix(
    float x, float y, float z,
    float sx, float sy, float sz,
    float yaw, float aspect,
    float camera_x, float camera_z)
{
    float c = cosf(yaw);
    float s = sinf(yaw);
    float f = 1.0f / tanf(0.5f);
    float near_z = 0.1f, far_z = 100.0f;
    float fz = far_z / (near_z - far_z);

    Matrix p = {{
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, fz, -1,
        0, 0, near_z * fz, 0
    }};

    Matrix m = {{
        sx * c, 0, -sx * s, 0,
        0, sy, 0, 0,
        sz * s, 0, sz * c, 0,
        x, y, z, 1
    }};

    Matrix view = {{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -camera_x, 0, -8.0f - camera_z, 1
    }};

    Matrix vm = multiply(&view, &m);
    return multiply(&p, &vm);
}

static float axis(int16_t v)
{
    return (v > STICK_DEAD_ZONE || v < -STICK_DEAD_ZONE)
        ? (float)v / 32768.0f : 0.0f;
}

static void write_gpu(volatile uint32_t *dst, const void *src,
                      uint32_t bytes)
{
    for (uint32_t i = 0; i < bytes / 4; i++) {
        uint32_t word;
        memcpy(&word, (const uint8_t *)src + i * 4, 4);
        dst[i] = word;
    }
    xsdk_gpu_invalidate(xsdk_gpu_address(dst), bytes);
}

static bool make_mesh(Game *g)
{
    float v[24][8];
    uint16_t idx[36];

    static const float pos[8][3] = {
        {-0.5f,-0.5f,-0.5f}, {0.5f,-0.5f,-0.5f},
        {-0.5f, 0.5f,-0.5f}, {0.5f, 0.5f,-0.5f},
        {-0.5f,-0.5f, 0.5f}, {0.5f,-0.5f, 0.5f},
        {-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}
    };

    static const uint8_t faces[6][4] = {
        {1,5,3,7}, {4,0,6,2}, {2,3,6,7},
        {0,1,4,5}, {5,4,7,6}, {0,2,1,3}
    };

    static const float colors[6][3] = {
        {1,0.2f,0.2f}, {0.2f,1,0.2f},
        {0.2f,0.4f,1}, {1,1,0.2f},
        {1,0.3f,1}, {0.2f,1,1}
    };

    for (int f = 0; f < 6; f++) {
        for (int c = 0; c < 4; c++) {
            int n = faces[f][c];
            for (int k = 0; k < 3; k++) {
                v[f * 4 + c][k] = pos[n][k];
                v[f * 4 + c][3 + k] = colors[f][k];
            }
            v[f * 4 + c][6] = (c & 1) ? 1.0f : 0.0f;
            v[f * 4 + c][7] = (c & 2) ? 0.0f : 1.0f;
        }
        uint16_t b = (uint16_t)(f * 4);
        uint16_t q[6] = {b, b+1, b+2, b+2, b+1, b+3};
        memcpy(&idx[f * 6], q, sizeof q);
    }

    g->vertices = xsdk_gpu_alloc(VERTICES_BYTES);
    g->indices = xsdk_gpu_alloc(INDICES_BYTES);
    if (!g->vertices || !g->indices) return false;

    write_gpu(g->vertices, v, VERTICES_BYTES);
    write_gpu(g->indices, idx, INDICES_BYTES);
    return true;
}

static bool init_game(Game *g)
{
    NTSTATUS s = xsdk_shader_load(
        &g->vertex, cube_vert.container, cube_vert.container_bytes);

    if (NT_SUCCESS(s))
        s = xsdk_shader_load(
            &g->pixel, cube_frag.container, cube_frag.container_bytes);

    if (NT_SUCCESS(s))
        s = xsdk_shader_set_vertex_layout(
            &g->vertex, layout, 3, (const uint32_t[]){STRIDE_BYTES});

    if (!NT_SUCCESS(s)) return false;

    XsdkDepthStencil d = {
        .depth_test = true,
        .depth_write = true,
        .depth_compare = XSDK_COMPARE_LESS_EQUAL
    };
    xsdk_gpu_depth_stencil_state(&g->depth, &d);

    if (!make_mesh(g)) return false;

    s = xsdk_texture_create(
        &g->texture, XSDK_FORMAT_A8R8G8B8, 2, 2, 1, true);

    if (!NT_SUCCESS(s)) return false;

    static const uint32_t texels[2][2] = {
        {0xFFFFFFFF, 0xFFFFFFFF},
        {0xFFFFFFFF, 0xFFFFFFFF}
    };

    xsdk_texture_upload(&g->texture, 0, texels, sizeof texels[0]);
    xsdk_gpu_invalidate(g->texture.address, g->texture.size_bytes);

    XsdkSampler sm = {
        .mag_filter = XSDK_FILTER_LINEAR,
        .min_filter = XSDK_FILTER_LINEAR,
        .mip_filter = XSDK_FILTER_POINT,
        .address_u = XSDK_ADDRESS_WRAP,
        .address_v = XSDK_ADDRESS_WRAP
    };

    xsdk_texture_sampler_state(&g->sampler, &sm);
    return true;
}

static void draw_object(Game *g, const Matrix *m)
{
    xsdk_gpu_set_vertex_constants(
        CUBE_VERT_PUSH, m->m, CUBE_VERT_PUSH_REGISTERS);

    xsdk_gpu_draw_indexed(
        XSDK_PRIMITIVE_TRIANGLE_LIST,
        xsdk_gpu_address(g->indices), 36, false);
}

static void draw_player_part(
    Game *g, float px, float foot_y, float pz,
    float ox, float oy, float oz,
    float sx, float sy, float sz, float yaw,
    float aspect, float camera_x, float camera_z)
{
    float c = cosf(yaw);
    float s = sinf(yaw);
    float wx = px + ox * c + oz * s;
    float wz = pz - ox * s + oz * c;

    Matrix m = model_matrix(
        wx, foot_y + oy, wz, sx, sy, sz,
        yaw, aspect, camera_x, camera_z);

    draw_object(g, &m);
}

static void draw_player(
    Game *g, float x, float foot_y, float z, float yaw,
    float aspect, float camera_x, float camera_z)
{
    /* Pernas */
    draw_player_part(g, x, foot_y, z, -0.105f, 0.12f, 0,
        0.12f, 0.24f, 0.14f, yaw, aspect, camera_x, camera_z);

    draw_player_part(g, x, foot_y, z, 0.105f, 0.12f, 0,
        0.12f, 0.24f, 0.14f, yaw, aspect, camera_x, camera_z);

    /* Tronco */
    draw_player_part(g, x, foot_y, z, 0, 0.39f, 0,
        0.31f, 0.34f, 0.19f, yaw, aspect, camera_x, camera_z);

    /* Braços */
    draw_player_part(g, x, foot_y, z, -0.225f, 0.39f, 0,
        0.10f, 0.30f, 0.12f, yaw, aspect, camera_x, camera_z);

    draw_player_part(g, x, foot_y, z, 0.225f, 0.39f, 0,
        0.10f, 0.30f, 0.12f, yaw, aspect, camera_x, camera_z);

    /* Cabeça */
    draw_player_part(g, x, foot_y, z, 0, 0.69f, 0,
        0.25f, 0.25f, 0.25f, yaw, aspect, camera_x, camera_z);
}

int main(void)
{
    xsdk_log_write(
        "GG Fall: left stick moves, A jumps, Back exits");

    NTSTATUS s = xsdk_splash_show(1500);
    (void)s;

    s = xsdk_gpu_open();
    if (!NT_SUCCESS(s)) return 1;

    static Game g;
    if (!init_game(&g)) return 1;

    uint32_t w = 0, h = 0;
    s = xsdk_present_open(
        XSDK_DISPLAY_720P, BUFFERS, SWAP_INTERVAL, &w, &h);
    if (!NT_SUCCESS(s)) return 1;

    float x = 0.0f;
    float foot_y = -0.88f;
    float z = 0.0f;
    float vy = 0.0f;
    float yaw = 0.0f;
    bool grounded = true;

    for (;;) {
        INPUT_STATE in = {0};

        if (XamInputGetState(0, 0, &in) == ERROR_SUCCESS &&
            (in.gamepad.buttons & INPUT_GAMEPAD_BACK))
            break;

        float move_x = axis(in.gamepad.thumb_lx);
        /* Frente/trás corrigido */
        float move_z = -axis(in.gamepad.thumb_ly);

        x += move_x * SPEED;
        z += move_z * SPEED;

        /* Gira na direção do movimento */
        if (move_x != 0.0f || move_z != 0.0f)
            yaw = atan2f(move_x, move_z);

        if ((in.gamepad.buttons & INPUT_GAMEPAD_A) && grounded) {
            vy = JUMP_SPEED;
            grounded = false;
        }

        foot_y += vy;

        if (!grounded)
            vy -= GRAVITY;

        if (foot_y <= -0.88f) {
            foot_y = -0.88f;
            vy = 0.0f;
            grounded = true;
        }

        uint32_t address;
        s = xsdk_present_begin(&address, BEGIN_TIMEOUT_MS);
        if (!NT_SUCCESS(s)) break;

        float aspect = (float)w / (float)h;
        float camera_x = x * CAMERA_FOLLOW;
        float camera_z = z * CAMERA_FOLLOW;

        xsdk_gpu_set_target(w, h);
        xsdk_gpu_set_depth_stencil(&g.depth);
        xsdk_gpu_set_shaders(&g.vertex, &g.pixel);
        xsdk_built_shader_set_literals(&cube_vert);
        xsdk_built_shader_set_literals(&cube_frag);
        xsdk_gpu_set_texture(
            CUBE_FRAG_SAMPLER_0_0, &g.texture, &g.sampler);
        xsdk_gpu_set_vertices(
            0, xsdk_gpu_address(g.vertices), VERTICES_BYTES);

        /* Plataforma */
        Matrix platform = model_matrix(
            0.0f, -0.94f, 0.0f,
            5.0f, 0.12f, 4.0f,
            0.0f, aspect, camera_x, camera_z);
        draw_object(&g, &platform);

        /* Obstáculo 1 */
        Matrix obstacle1 = model_matrix(
            -1.1f, -0.68f, -1.25f,
            0.55f, 0.40f, 0.55f,
            0.0f, aspect, camera_x, camera_z);
        draw_object(&g, &obstacle1);

        /* Obstáculo 2 */
        Matrix obstacle2 = model_matrix(
            0.85f, -0.72f, -2.35f,
            0.75f, 0.32f, 0.50f,
            0.35f, aspect, camera_x, camera_z);
        draw_object(&g, &obstacle2);

        /* Obstáculo 3 */
        Matrix obstacle3 = model_matrix(
            -0.65f, -0.70f, 1.15f,
            0.45f, 0.36f, 0.45f,
            0.6f, aspect, camera_x, camera_z);
        draw_object(&g, &obstacle3);

        /* Boneco 3D */
        draw_player(
            &g, x, foot_y, z, yaw,
            aspect, camera_x, camera_z);

        xsdk_gpu_resolve(address, w, h, CLEAR_ARGB);
        xsdk_present_end();
    }

    xsdk_present_close();
    xsdk_texture_free(&g.texture);
    xsdk_gpu_free(g.indices);
    xsdk_gpu_free(g.vertices);
    xsdk_shader_free(&g.pixel);
    xsdk_shader_free(&g.vertex);
    xsdk_gpu_close();

    return 0;
}
