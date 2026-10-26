
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

#define DEAD_ZONE 8000
#define SPEED 0.035f
#define JUMP_SPEED 0.095f
#define GRAVITY 0.0042f
#define DASH_SPEED 0.105f
#define DASH_UP_SPEED 0.025f
#define DASH_DURATION 30

#define CAMERA_DISTANCE 5.5f
#define CAMERA_TARGET_HEIGHT 0.45f
#define CAMERA_TURN_SPEED 0.035f
#define CAMERA_PITCH_SPEED 0.025f
#define CAMERA_MIN_PITCH (-0.55f)
#define CAMERA_MAX_PITCH 0.65f

#define FLOOR_Y (-0.88f)
#define PLAYER_HEIGHT 0.80f
#define PLAYER_RADIUS 0.18f
#define FALL_LIMIT (-3.0f)

#define ARENA_MIN_X (-2.5f)
#define ARENA_MAX_X 2.5f
#define ARENA_MIN_Z (-10.0f)
#define ARENA_MAX_Z 3.0f

#define WALL_COUNT 5
#define WALL_INTERVAL 210
#define WALL_SPEED_START 0.018f
#define WALL_SPEED_MAX 0.040f
#define WALL_THICKNESS 0.32f
#define WALL_HEIGHT 1.25f
#define WALL_GAP_WIDTH 1.55f
#define SURVIVE_FRAMES 3600

typedef struct {
    float m[16];
} Matrix;

typedef struct {
    XsdkShader vertex, pixel;
    XsdkTexture texture;
    XsdkSamplerState sampler;
    XsdkDepthStencilState depth;
    volatile uint32_t *vertices, *indices;
} Game;

typedef struct {
    float x, y, z;
    float sx, sy, sz;
    float yaw;
} Obstacle;

typedef struct {
    float z;
    int gap;
    bool active;
} DashWall;

typedef enum {
    MENU_HOME,
    MENU_PLAYING,
    MENU_LOST,
    MENU_WON
} GameState;

static float camera_yaw = 0.0f;
static float camera_pitch = 0.20f;
static float camera_y = FLOOR_Y + CAMERA_TARGET_HEIGHT;

static const XsdkVertexElement layout[] = {
    {.usage = XSDK_USAGE_POSITION,
     .format = XSDK_VERTEX_FLOAT3, .offset_bytes = 0},
    {.usage = XSDK_USAGE_COLOR,
     .format = XSDK_VERTEX_FLOAT3, .offset_bytes = 12},
    {.usage = XSDK_USAGE_TEXCOORD,
     .format = XSDK_VERTEX_FLOAT2, .offset_bytes = 24}
};

static Matrix multiply(const Matrix *a, const Matrix *b)
{
    Matrix r;

    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;

            for (int k = 0; k < 4; k++)
                sum += a->m[k * 4 + row] *
                       b->m[c * 4 + k];

            r.m[c * 4 + row] = sum;
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
    float near_z = 0.1f;
    float far_z = 100.0f;
    float fz = far_z / (near_z - far_z);

    Matrix p = {{
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, fz, -1,
        0, 0, near_z * fz, 0
    }};

    Matrix m = {{
        sx*c, 0, -sx*s, 0,
        0, sy, 0, 0,
        sz*s, 0, sz*c, 0,
        x, y, z, 1
    }};

    Matrix target = {{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -camera_x, -camera_y, -camera_z, 1
    }};

    float cy = cosf(-camera_yaw);
    float syaw = sinf(-camera_yaw);

    Matrix ry = {{
        cy, 0, -syaw, 0,
        0, 1, 0, 0,
        syaw, 0, cy, 0,
        0, 0, 0, 1
    }};

    float cp = cosf(-camera_pitch);
    float sp = sinf(-camera_pitch);

    Matrix rx = {{
        1, 0, 0, 0,
        0, cp, -sp, 0,
        0, sp, cp, 0,
        0, 0, 0, 1
    }};

    Matrix distance = {{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, -CAMERA_DISTANCE, 1
    }};

    Matrix view = multiply(&ry, &target);
    view = multiply(&rx, &view);
    view = multiply(&distance, &view);

    Matrix vm = multiply(&view, &m);
    return multiply(&p, &vm);
}

static float axis(int16_t v)
{
    return (v > DEAD_ZONE || v < -DEAD_ZONE)
        ? (float)v / 32768.0f : 0.0f;
}

static void write_gpu(
    volatile uint32_t *dst,
    const void *src,
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
        {-0.5f,0.5f,-0.5f},  {0.5f,0.5f,-0.5f},
        {-0.5f,-0.5f,0.5f},  {0.5f,-0.5f,0.5f},
        {-0.5f,0.5f,0.5f},   {0.5f,0.5f,0.5f}
    };

    static const uint8_t faces[6][4] = {
        {1,5,3,7}, {4,0,6,2}, {2,3,6,7},
        {0,1,4,5}, {5,4,7,6}, {0,2,1,3}
    };

    static const float colors[6][3] = {
        {1.0f,0.2f,0.2f}, {0.2f,1.0f,0.2f},
        {0.2f,0.4f,1.0f}, {1.0f,1.0f,0.2f},
        {1.0f,0.3f,1.0f}, {0.2f,1.0f,1.0f}
    };

    for (int f = 0; f < 6; f++) {
        for (int c = 0; c < 4; c++) {
            int n = faces[f][c];

            for (int k = 0; k < 3; k++) {
                v[f*4+c][k] = pos[n][k];
                v[f*4+c][3+k] = colors[f][k];
            }

            v[f*4+c][6] = (c & 1) ? 1.0f : 0.0f;
            v[f*4+c][7] = (c & 2) ? 0.0f : 1.0f;
        }

        uint16_t b = (uint16_t)(f * 4);
        uint16_t q[6] = {b,b+1,b+2,b+2,b+1,b+3};
        memcpy(&idx[f*6], q, sizeof q);
    }

    g->vertices = xsdk_gpu_alloc(VERTICES_BYTES);
    g->indices = xsdk_gpu_alloc(INDICES_BYTES);

    if (!g->vertices || !g->indices)
        return false;

    write_gpu(g->vertices, v, VERTICES_BYTES);
    write_gpu(g->indices, idx, INDICES_BYTES);
    return true;
}

static bool init_game(Game *g)
{
    NTSTATUS s = xsdk_shader_load(
        &g->vertex, cube_vert.container,
        cube_vert.container_bytes);

    if (NT_SUCCESS(s))
        s = xsdk_shader_load(
            &g->pixel, cube_frag.container,
            cube_frag.container_bytes);

    if (NT_SUCCESS(s))
        s = xsdk_shader_set_vertex_layout(
            &g->vertex, layout, 3,
            (const uint32_t[]){STRIDE_BYTES});

    if (!NT_SUCCESS(s))
        return false;

    XsdkDepthStencil d = {
        .depth_test = true,
        .depth_write = true,
        .depth_compare = XSDK_COMPARE_LESS_EQUAL
    };

    xsdk_gpu_depth_stencil_state(&g->depth, &d);

    if (!make_mesh(g))
        return false;

    s = xsdk_texture_create(
        &g->texture, XSDK_FORMAT_A8R8G8B8,
        2, 2, 1, true);

    if (!NT_SUCCESS(s))
        return false;

    static const uint32_t texels[2][2] = {
        {0xFFFFFFFF,0xFFFFFFFF},
        {0xFFFFFFFF,0xFFFFFFFF}
    };

    xsdk_texture_upload(
        &g->texture, 0, texels, sizeof texels[0]);

    xsdk_gpu_invalidate(
        g->texture.address, g->texture.size_bytes);

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

static void draw_player(
    Game *g, float x, float y, float z,
    float yaw, float aspect,
    float cx, float cz, float phase,
    bool moving, bool diving)
{
    float swing = moving ? sinf(phase) * 0.07f : 0.0f;

    float body_y = diving ? 0.32f : 0.39f;
    float body_h = diving ? 0.16f : 0.34f;
    float body_z = diving ? 0.0f : 0.0f;
    float body_sz = diving ? 0.48f : 0.19f;

    Matrix body = model_matrix(
        x, y + body_y, z + body_z,
        diving ? 0.28f : 0.31f,
        body_h, body_sz, yaw, aspect, cx, cz);
    draw_object(g, &body);

    if (!diving) {
        Matrix head = model_matrix(
            x, y + 0.69f, z,
            0.25f, 0.25f, 0.25f,
            yaw, aspect, cx, cz);
        draw_object(g, &head);
    }

    for (int side = -1; side <= 1; side += 2) {
        float leg_z = side < 0 ? swing : -swing;
        float leg_x = side * 0.105f;

        Matrix leg = model_matrix(
            x + leg_x, y + (diving ? 0.27f : 0.12f),
            z + (diving ? -0.30f : leg_z),
            0.12f, diving ? 0.12f : 0.24f,
            diving ? 0.25f : 0.14f,
            yaw, aspect, cx, cz);
        draw_object(g, &leg);

        Matrix arm = model_matrix(
            x + side * 0.225f,
            y + (diving ? 0.35f : 0.39f),
            z + (diving ? 0.25f : -leg_z),
            0.10f, diving ? 0.10f : 0.30f,
            diving ? 0.36f : 0.12f,
            yaw, aspect, cx, cz);
        draw_object(g, &arm);
    }
}

static void draw_floor(
    Game *g, float aspect, float cx, float cz)
{
    Matrix floor = model_matrix(
        0.0f, FLOOR_Y - 0.06f, -3.5f,
        5.0f, 0.12f, 13.0f,
        0.0f, aspect, cx, cz);
    draw_object(g, &floor);

    /* Bordas laterais da arena. */
    for (int side = -1; side <= 1; side += 2) {
        Matrix rail = model_matrix(
            side * 2.55f, FLOOR_Y + 0.25f, -3.5f,
            0.12f, 0.50f, 13.0f,
            0.0f, aspect, cx, cz);
        draw_object(g, &rail);
    }
}

static DashWall walls[WALL_COUNT];
static int wall_timer = 0;
static unsigned wall_index = 0;
static int survival_time = 0;
static float wall_speed = WALL_SPEED_START;

static void reset_walls(void)
{
    for (int i = 0; i < WALL_COUNT; i++) {
        walls[i].z = 30.0f;
        walls[i].gap = i % 3;
        walls[i].active = false;
    }

    wall_timer = 0;
    wall_index = 0;
    survival_time = 0;
    wall_speed = WALL_SPEED_START;
}

static void spawn_wall(void)
{
    unsigned i = wall_index++ % WALL_COUNT;

    walls[i].z = 3.0f;
    walls[i].gap = (int)((wall_index + survival_time / 300) % 3);
    walls[i].active = true;
}

static void update_walls(void)
{
    wall_timer++;
    survival_time++;

    if (wall_timer >= WALL_INTERVAL) {
        wall_timer = 0;
        spawn_wall();
    }

    /* A velocidade aumenta gradualmente. */
    wall_speed = WALL_SPEED_START +
        (float)survival_time * 0.000004f;

    if (wall_speed > WALL_SPEED_MAX)
        wall_speed = WALL_SPEED_MAX;

    for (int i = 0; i < WALL_COUNT; i++) {
        if (!walls[i].active)
            continue;

        walls[i].z -= wall_speed;

        if (walls[i].z < ARENA_MIN_Z - 1.0f)
            walls[i].active = false;
    }
}

static void draw_walls(
    Game *g, float aspect, float cx, float cz)
{
    for (int i = 0; i < WALL_COUNT; i++) {
        if (!walls[i].active)
            continue;

        for (int lane = 0; lane < 3; lane++) {
            if (lane == walls[i].gap)
                continue;

            float segment_x = -1.65f + lane * 1.65f;

            Matrix segment = model_matrix(
                segment_x,
                FLOOR_Y + WALL_HEIGHT * 0.5f,
                walls[i].z,
                1.65f, WALL_HEIGHT, WALL_THICKNESS,
                0.0f, aspect, cx, cz);

            draw_object(g, &segment);
        }
    }
}

static bool hit_wall(float x, float y, float z)
{
    for (int i = 0; i < WALL_COUNT; i++) {
        if (!walls[i].active)
            continue;

        if (fabsf(z - walls[i].z) >
            WALL_THICKNESS * 0.5f + PLAYER_RADIUS)
            continue;

        if (y >= FLOOR_Y + WALL_HEIGHT)
            continue;

        for (int lane = 0; lane < 3; lane++) {
            if (lane == walls[i].gap)
                continue;

            float center_x = -1.65f + lane * 1.65f;

            if (fabsf(x - center_x) <
                0.825f + PLAYER_RADIUS)
                return true;
        }
    }

    return false;
}

static void draw_menu_scene(
    Game *g, float aspect)
{
    Matrix floor = model_matrix(
        0.0f, -0.94f, 0.0f,
        4.5f, 0.12f, 4.5f,
        0.0f, aspect, 0.0f, 0.0f);
    draw_object(g, &floor);

    /* Blocos decorativos do menu. */
    Matrix left = model_matrix(
        -1.15f, -0.25f, 0.0f,
        0.55f, 0.55f, 0.55f,
        0.2f, aspect, 0.0f, 0.0f);
    draw_object(g, &left);

    Matrix center = model_matrix(
        0.0f, 0.0f, 0.0f,
        0.75f, 0.75f, 0.75f,
        0.0f, aspect, 0.0f, 0.0f);
    draw_object(g, &center);

    Matrix right = model_matrix(
        1.15f, -0.25f, 0.0f,
        0.55f, 0.55f, 0.55f,
        -0.2f, aspect, 0.0f, 0.0f);
    draw_object(g, &right);
}

int main(void)
{
    xsdk_log_write("GG Fall: Block Dash");

    xsdk_splash_show(1500);

    NTSTATUS s = xsdk_gpu_open();
    if (!NT_SUCCESS(s))
        return 1;

    static Game g;

    if (!init_game(&g))
        return 1;

    uint32_t w = 0, h = 0;

    s = xsdk_present_open(
        XSDK_DISPLAY_720P,
        BUFFERS,
        SWAP_INTERVAL,
        &w, &h);

    if (!NT_SUCCESS(s))
        return 1;

    GameState state = MENU_HOME;

    float x = 0.0f;
    float foot_y = FLOOR_Y;
    float z = 0.0f;
    float vy = 0.0f;
    float yaw = 0.0f;
    float walk_phase = 0.0f;

    float dash_vx = 0.0f;
    float dash_vz = 0.0f;
    int dash_timer = 0;

    bool grounded = true;
    bool has_dashed = false;
    bool previous_a = false;

    for (;;) {
        INPUT_STATE in = {0};

        XamInputGetState(0, 0, &in);

        if (in.gamepad.buttons & INPUT_GAMEPAD_BACK)
            break;

        bool a_down =
            (in.gamepad.buttons & INPUT_GAMEPAD_A) != 0;

        bool a_pressed = a_down && !previous_a;
        previous_a = a_down;

        float look_x = axis(in.gamepad.thumb_rx);
        float look_y = axis(in.gamepad.thumb_ry);

        float stick_x = axis(in.gamepad.thumb_lx);
        float stick_z = -axis(in.gamepad.thumb_ly);

        camera_yaw -= look_x * CAMERA_TURN_SPEED;
        camera_pitch -= look_y * CAMERA_PITCH_SPEED;

        if (camera_pitch < CAMERA_MIN_PITCH)
            camera_pitch = CAMERA_MIN_PITCH;

        if (camera_pitch > CAMERA_MAX_PITCH)
            camera_pitch = CAMERA_MAX_PITCH;

        float cy = cosf(camera_yaw);
        float sy = sinf(camera_yaw);

        float move_x = stick_x * cy + stick_z * sy;
        float move_z = -stick_x * sy + stick_z * cy;

        bool moving =
            stick_x != 0.0f || stick_z != 0.0f;

        if (moving)
            walk_phase += 0.22f;

        /*
         * MENU:
         * Aperte A para começar.
         */
        if (state == MENU_HOME) {
            if (a_pressed) {
                x = 0.0f;
                foot_y = FLOOR_Y;
                z = 0.0f;
                vy = 0.0f;
                yaw = 0.0f;

                dash_timer = 0;
                has_dashed = false;
                grounded = true;

                camera_yaw = 0.0f;
                camera_pitch = 0.20f;

                reset_walls();
                state = MENU_PLAYING;

                xsdk_log_write("GG Fall: start Block Dash");
            }
        }

        /*
         * DERROTA OU VITÓRIA:
         * Aperte A para voltar ao menu.
         */
        else if (state == MENU_LOST ||
                 state == MENU_WON) {
            if (a_pressed) {
                state = MENU_HOME;
                xsdk_log_write("GG Fall: back to menu");
            }
        }

        /*
         * JOGO:
         * movimento, pulo, dash e paredes.
         */
        else if (state == MENU_PLAYING) {
            if (dash_timer > 0) {
                float nx = x + dash_vx;
                float nz = z + dash_vz;

                if (nx >= ARENA_MIN_X &&
                    nx <= ARENA_MAX_X)
                    x = nx;

                if (nz >= ARENA_MIN_Z &&
                    nz <= ARENA_MAX_Z)
                    z = nz;

                dash_timer--;
            } else {
                float nx = x + move_x * SPEED;
                float nz = z + move_z * SPEED;

                if (nx >= ARENA_MIN_X &&
                    nx <= ARENA_MAX_X)
                    x = nx;

                if (nz >= ARENA_MIN_Z &&
                    nz <= ARENA_MAX_Z)
                    z = nz;
            }

            if (moving && dash_timer == 0)
                yaw = atan2f(move_x, move_z);

            if (a_pressed && grounded) {
                vy = JUMP_SPEED;
                grounded = false;
                has_dashed = false;
            } else if (a_pressed && !grounded &&
                       !has_dashed) {
                has_dashed = true;
                dash_timer = DASH_DURATION;

                float len = sqrtf(
                    move_x * move_x +
                    move_z * move_z);

                if (len > 0.001f) {
                    dash_vx = move_x / len * DASH_SPEED;
                    dash_vz = move_z / len * DASH_SPEED;
                } else {
                    dash_vx = sinf(yaw) * DASH_SPEED;
                    dash_vz = cosf(yaw) * DASH_SPEED;
                }

                vy = DASH_UP_SPEED;
            }

            float previous_y = foot_y;
            foot_y += vy;

            if (!grounded)
                vy -= GRAVITY;

            if (vy <= 0.0f &&
                previous_y >= FLOOR_Y &&
                foot_y <= FLOOR_Y) {
                foot_y = FLOOR_Y;
                vy = 0.0f;
                grounded = true;
                has_dashed = false;
            }

            if (foot_y < FALL_LIMIT) {
                state = MENU_LOST;
                xsdk_log_write("GG Fall: fell off the arena");
            }

            update_walls();

            if (hit_wall(x, foot_y, z)) {
                state = MENU_LOST;
                xsdk_log_write("GG Fall: hit a wall");
            }

            if (survival_time >= SURVIVE_FRAMES) {
                state = MENU_WON;
                xsdk_log_write("GG Fall: Block Dash completed");
            }
        }

        camera_y = foot_y + CAMERA_TARGET_HEIGHT;

        uint32_t address;

        s = xsdk_present_begin(
            &address, BEGIN_TIMEOUT_MS);

        if (!NT_SUCCESS(s))
            break;

        float aspect = (float)w / (float)h;

        float camera_x = x;
        float camera_z = z;

        xsdk_gpu_set_target(w, h);
        xsdk_gpu_set_depth_stencil(&g.depth);
        xsdk_gpu_set_shaders(&g.vertex, &g.pixel);

        xsdk_built_shader_set_literals(&cube_vert);
        xsdk_built_shader_set_literals(&cube_frag);

        xsdk_gpu_set_texture(
            CUBE_FRAG_SAMPLER_0_0,
            &g.texture, &g.sampler);

        xsdk_gpu_set_vertices(
            0,
            xsdk_gpu_address(g.vertices),
            VERTICES_BYTES);

        if (state == MENU_HOME) {
            camera_y = 0.0f;
            draw_menu_scene(&g, aspect);
        } else {
            draw_floor(&g, aspect, camera_x, camera_z);

            if (state == MENU_PLAYING)
                draw_walls(
                    &g, aspect, camera_x, camera_z);

            draw_player(
                &g, x, foot_y, z, yaw,
                aspect, camera_x, camera_z,
                walk_phase, moving,
                dash_timer > 0);
        }

        xsdk_gpu_resolve(
            address, w, h, CLEAR_ARGB);

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
