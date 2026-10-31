#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "GG-Fall_shaders.h"
#include "capybara_mesh.h"
#include "capybara_texture_data.h"
#include "capybara_rig.h"
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

#define CAPYBARA_VERTEX_BYTES \
    (CAPYBARA_VERTEX_COUNT * STRIDE_BYTES)
#define CAPYBARA_INDEX_BYTES \
    (CAPYBARA_INDEX_COUNT * 2)

#define STICK_DEAD_ZONE 8000
#define SPEED 0.035f
#define JUMP_SPEED 0.095f
#define GRAVITY 0.0042f
#define CAMERA_FOLLOW 1.0f

#define DASH_SPEED 0.105f
#define DASH_UP_SPEED 0.025f
#define DASH_GRAVITY 0.0042f
#define DASH_DURATION 30
#define SLIDE_DURATION 240
#define SLIDE_FRICTION 0.94f
#define SLIDE_STOP_SPEED 0.002f

#define CAMERA_TURN_SPEED 0.035f
#define CAMERA_PITCH_SPEED 0.025f
#define CAMERA_MIN_PITCH (-0.55f)
#define CAMERA_MAX_PITCH (0.65f)
#define CAMERA_DISTANCE 5.5f
#define CAMERA_TARGET_HEIGHT 0.45f

#define FLOOR_Y (-0.88f)
#define PLAYER_HEIGHT 0.80f
#define PLAYER_RADIUS 0.18f

#define PLATFORM_MIN_X (-2.5f)
#define PLATFORM_MAX_X (2.5f)
#define PLATFORM_MIN_Z (-9.0f)
#define PLATFORM_MAX_Z (2.0f)
#define FALL_LIMIT (-3.0f)
#define CHECKPOINT_RADIUS 0.55f
#define FINISH_Z (-8.20f)

typedef struct {
    float m[16];
} Matrix;

typedef struct {
    XsdkShader vertex, pixel;
    XsdkTexture texture;
    XsdkSamplerState sampler;
    XsdkDepthStencilState depth;

    volatile uint32_t *vertices, *indices;

    volatile uint32_t *capybara_vertices_gpu;
    volatile uint32_t *capybara_indices_gpu;

    XsdkTexture capybara_texture;
    XsdkSamplerState capybara_sampler;
} Game;

typedef struct {
    float x, y, z;
    float sx, sy, sz;
    float yaw;
} Obstacle;

typedef struct {
    float x, z;
} Checkpoint;

static float camera_yaw = 0.0f;
static float camera_pitch = 0.20f;
static float camera_y = FLOOR_Y + CAMERA_TARGET_HEIGHT;

static const Obstacle obstacles[] = {
    {-1.10f, -0.68f, -1.25f,
     0.55f, 0.40f, 0.55f, 0.0f},

    {0.85f, -0.72f, -2.35f,
     0.75f, 0.32f, 0.50f, 0.35f},

    {-0.65f, -0.70f, 1.15f,
     0.45f, 0.36f, 0.45f, 0.6f},

    {1.35f, -0.65f, -4.35f,
     0.65f, 0.46f, 0.65f, 0.0f},

    {-0.85f, -0.70f, -5.45f,
     0.70f, 0.36f, 0.55f, 0.2f},

    {0.45f, -0.62f, -6.65f,
     0.60f, 0.52f, 0.60f, 0.0f},

    {-1.45f, -0.72f, -7.45f,
     0.50f, 0.32f, 0.50f, 0.0f}
};

#define OBSTACLE_COUNT \
    (sizeof(obstacles) / sizeof(obstacles[0]))

static const Checkpoint checkpoints[] = {
    {1.55f, -0.25f},
    {-1.55f, -3.30f},
    {0.00f, -5.95f}
};

#define CHECKPOINT_COUNT \
    (sizeof(checkpoints) / sizeof(checkpoints[0]))

static const XsdkVertexElement layout[] = {
    {
        .usage = XSDK_USAGE_POSITION,
        .format = XSDK_VERTEX_FLOAT3,
        .offset_bytes = 0
    },
    {
        .usage = XSDK_USAGE_COLOR,
        .format = XSDK_VERTEX_FLOAT3,
        .offset_bytes = 12
    },
    {
        .usage = XSDK_USAGE_TEXCOORD,
        .format = XSDK_VERTEX_FLOAT2,
        .offset_bytes = 24
    }
};

static Matrix multiply(const Matrix *a, const Matrix *b)
{
    Matrix r;

    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;

            for (int k = 0; k < 4; k++) {
                sum += a->m[k * 4 + row] *
                       b->m[c * 4 + k];
            }

            r.m[c * 4 + row] = sum;
        }
    }

    return r;
}

/*
 * Câmera orbital:
 * acompanha o jogador, gira horizontalmente
 * e verticalmente e aplica a distância da câmera.
 */
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
        sx * c, 0, -sx * s, 0,
        0, sy, 0, 0,
        sz * s, 0, sz * c, 0,
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

    Matrix rotate_y = {{
        cy, 0, -syaw, 0,
        0, 1, 0, 0,
        syaw, 0, cy, 0,
        0, 0, 0, 1
    }};

    float cp = cosf(-camera_pitch);
    float sp = sinf(-camera_pitch);

    Matrix rotate_x = {{
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

    Matrix view = multiply(&rotate_y, &target);
    view = multiply(&rotate_x, &view);
    view = multiply(&distance, &view);

    Matrix vm = multiply(&view, &m);
    return multiply(&p, &vm);
}

static float axis(int16_t v)
{
    return (v > STICK_DEAD_ZONE ||
            v < -STICK_DEAD_ZONE)
        ? (float)v / 32768.0f
        : 0.0f;
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
        {-0.5f, -0.5f, -0.5f},
        { 0.5f, -0.5f, -0.5f},
        {-0.5f,  0.5f, -0.5f},
        { 0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f,  0.5f},
        { 0.5f, -0.5f,  0.5f},
        {-0.5f,  0.5f,  0.5f},
        { 0.5f,  0.5f,  0.5f}
    };

    static const uint8_t faces[6][4] = {
        {1, 5, 3, 7},
        {4, 0, 6, 2},
        {2, 3, 6, 7},
        {0, 1, 4, 5},
        {5, 4, 7, 6},
        {0, 2, 1, 3}
    };

    static const float colors[6][3] = {
        {1, 0.2f, 0.2f},
        {0.2f, 1, 0.2f},
        {0.2f, 0.4f, 1},
        {1, 1, 0.2f},
        {1, 0.3f, 1},
        {0.2f, 1, 1}
    };

    for (int f = 0; f < 6; f++) {
        for (int c = 0; c < 4; c++) {
            int n = faces[f][c];

            for (int k = 0; k < 3; k++) {
                v[f * 4 + c][k] = pos[n][k];
                v[f * 4 + c][3 + k] = colors[f][k];
            }

            v[f * 4 + c][6] =
                (c & 1) ? 1.0f : 0.0f;
            v[f * 4 + c][7] =
                (c & 2) ? 0.0f : 1.0f;
        }

        uint16_t b = (uint16_t)(f * 4);
        uint16_t q[6] = {
            b, b + 1, b + 2,
            b + 2, b + 1, b + 3
        };

        memcpy(&idx[f * 6], q, sizeof q);
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
        &g->vertex,
        cube_vert.container,
        cube_vert.container_bytes);

    if (NT_SUCCESS(s)) {
        s = xsdk_shader_load(
            &g->pixel,
            cube_frag.container,
            cube_frag.container_bytes);
    }

    if (NT_SUCCESS(s)) {
        s = xsdk_shader_set_vertex_layout(
            &g->vertex,
            layout,
            3,
            (const uint32_t[]){STRIDE_BYTES});
    }

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

    /* Cria os buffers do modelo da capivara. */
    g->capybara_vertices_gpu =
        xsdk_gpu_alloc(CAPYBARA_VERTEX_BYTES);

    g->capybara_indices_gpu =
        xsdk_gpu_alloc(CAPYBARA_INDEX_BYTES);

    if (!g->capybara_vertices_gpu ||
        !g->capybara_indices_gpu)
        return false;

    write_gpu(
        g->capybara_vertices_gpu,
        capybara_vertices,
        CAPYBARA_VERTEX_BYTES);

    write_gpu(
        g->capybara_indices_gpu,
        capybara_indices,
        CAPYBARA_INDEX_BYTES);

    /* Textura branca usada no cenário e nos obstáculos. */
    s = xsdk_texture_create(
        &g->texture,
        XSDK_FORMAT_A8R8G8B8,
        2, 2, 1, true);

    if (!NT_SUCCESS(s))
        return false;

    static const uint32_t texels[2][2] = {
        {0xFFFFFFFF, 0xFFFFFFFF},
        {0xFFFFFFFF, 0xFFFFFFFF}
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

    /* Textura extraída do modelo da capivara. */
    s = xsdk_texture_create(
        &g->capybara_texture,
        XSDK_FORMAT_A8R8G8B8,
        CAPYBARA_TEXTURE_WIDTH,
        CAPYBARA_TEXTURE_HEIGHT,
        1, true);

    if (!NT_SUCCESS(s))
        return false;

    xsdk_texture_upload(
        &g->capybara_texture,
        0,
        capybara_texels,
        sizeof capybara_texels[0]);

    xsdk_gpu_invalidate(
        g->capybara_texture.address,
        g->capybara_texture.size_bytes);

    xsdk_texture_sampler_state(
        &g->capybara_sampler, &sm);

    return true;
}

static void draw_object(Game *g, const Matrix *m)
{
    xsdk_gpu_set_vertex_constants(
        CUBE_VERT_PUSH,
        m->m,
        CUBE_VERT_PUSH_REGISTERS);

    xsdk_gpu_draw_indexed(
        XSDK_PRIMITIVE_TRIANGLE_LIST,
        xsdk_gpu_address(g->indices),
        36,
        false);
}


/* Animação esquelética da capivara calculada na CPU. */
static Matrix rig_identity(void)
{
    Matrix m = {{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}};
    return m;
}

static Matrix rig_local_matrix(const CapybaraRigNode *n,
                               const float t[3],
                               const float q[4],
                               const float sc[3])
{
    float x=q[0], y=q[1], z=q[2], w=q[3];
    Matrix m = {{
        (1-2*(y*y+z*z))*sc[0], (2*(x*y+z*w))*sc[0], (2*(x*z-y*w))*sc[0], 0,
        (2*(x*y-z*w))*sc[1], (1-2*(x*x+z*z))*sc[1], (2*(y*z+x*w))*sc[1], 0,
        (2*(x*z+y*w))*sc[2], (2*(y*z-x*w))*sc[2], (1-2*(x*x+y*y))*sc[2], 0,
        t[0], t[1], t[2], 1
    }};
    (void)n;
    return m;
}

static void rig_sample_channel(const CapybaraAnimChannel *ch,
                               const CapybaraAnimKey *keys,
                               float time, float out[4])
{
    const CapybaraAnimKey *first = &keys[ch->first_key];
    if (ch->key_count <= 1 || time <= first->time) {
        memcpy(out, first->value, sizeof(float)*4);
        return;
    }
    const CapybaraAnimKey *last = &keys[ch->first_key + ch->key_count - 1];
    if (time >= last->time) {
        memcpy(out, last->value, sizeof(float)*4);
        return;
    }
    for (uint16_t i=0; i+1<ch->key_count; ++i) {
        const CapybaraAnimKey *a=&keys[ch->first_key+i];
        const CapybaraAnimKey *b=&keys[ch->first_key+i+1];
        if (time >= a->time && time <= b->time) {
            float d=b->time-a->time;
            float f=d>0 ? (time-a->time)/d : 0;
            if (ch->path == 1) {
                float dot=0;
                for(int k=0;k<4;k++) dot+=a->value[k]*b->value[k];
                float bx=b->value[0], by=b->value[1], bz=b->value[2], bw=b->value[3];
                if(dot<0){dot=-dot;bx=-bx;by=-by;bz=-bz;bw=-bw;}
                if(dot>0.9995f) {
                    for(int k=0;k<4;k++) out[k]=a->value[k]+f*((k==0?bx:k==1?by:k==2?bz:bw)-a->value[k]);
                } else {
                    float theta=acosf(dot), sn=sinf(theta);
                    float wa=sinf((1-f)*theta)/sn, wb=sinf(f*theta)/sn;
                    out[0]=wa*a->value[0]+wb*bx; out[1]=wa*a->value[1]+wb*by;
                    out[2]=wa*a->value[2]+wb*bz; out[3]=wa*a->value[3]+wb*bw;
                }
                float len=sqrtf(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]+out[3]*out[3]);
                if(len>0.00001f) for(int k=0;k<4;k++) out[k]/=len;
            } else {
                for(int k=0;k<4;k++) out[k]=a->value[k]+(b->value[k]-a->value[k])*f;
            }
            return;
        }
    }
    memcpy(out,last->value,sizeof(float)*4);
}

static void animate_capybara(Game *g, bool running, float dt)
{
    static float anim_time=0.0f;
    static float verts[CAPYBARA_RIG_VERTEX_COUNT * VERTEX_FLOATS];
    Matrix node_world[CAPYBARA_NODE_COUNT];
    Matrix skin[CAPYBARA_RIG_JOINT_COUNT];
    float tr[CAPYBARA_NODE_COUNT][3];
    float rot[CAPYBARA_NODE_COUNT][4];
    float scl[CAPYBARA_NODE_COUNT][3];
    const CapybaraAnimChannel *channels = running ? capybara_run_channels : capybara_idle_channels;
    const CapybaraAnimKey *keys = running ? capybara_run_keys : capybara_idle_keys;
    unsigned channel_count = running ? 31u : 38u;
    float duration = running ? CAPYBARA_RUN_DURATION : CAPYBARA_IDLE_DURATION;
    anim_time += dt;
    if(duration>0) { while(anim_time>=duration) anim_time-=duration; }
    for(unsigned n=0;n<CAPYBARA_NODE_COUNT;n++) {
        memcpy(tr[n],capybara_rig_nodes[n].translation,sizeof tr[n]);
        memcpy(rot[n],capybara_rig_nodes[n].rotation,sizeof rot[n]);
        memcpy(scl[n],capybara_rig_nodes[n].scale,sizeof scl[n]);
    }
    for(unsigned c=0;c<channel_count;c++) {
        const CapybaraAnimChannel *ch=&channels[c];
        float v[4]; rig_sample_channel(ch,keys,anim_time,v);
        if(ch->path==0) memcpy(tr[ch->node],v,sizeof(float)*3);
        else if(ch->path==1) memcpy(rot[ch->node],v,sizeof(float)*4);
        else if(ch->path==2) memcpy(scl[ch->node],v,sizeof(float)*3);
    }
    for(unsigned n=0;n<CAPYBARA_NODE_COUNT;n++) {
        Matrix local=rig_local_matrix(&capybara_rig_nodes[n],tr[n],rot[n],scl[n]);
        int parent=capybara_rig_nodes[n].parent;
        node_world[n]=(parent>=0)?multiply(&node_world[parent],&local):local;
    }
    for(unsigned n=0;n<CAPYBARA_NODE_COUNT;n++) {
        int j=capybara_rig_nodes[n].joint_index;
        if(j>=0 && j<CAPYBARA_RIG_JOINT_COUNT) {
            Matrix ibm;
            memcpy(ibm.m,&capybara_inverse_bind_matrices[j*16],sizeof ibm.m);
            skin[j]=multiply(&node_world[n],&ibm);
        }
    }
    for(unsigned i=0;i<CAPYBARA_RIG_VERTEX_COUNT;i++) {
        float px=0,py=0,pz=0;
        for(unsigned k=0;k<4;k++) {
            float w=capybara_rig_weights[i*4+k];
            unsigned j=capybara_rig_joints[i*4+k];
            if(w<=0 || j>=CAPYBARA_RIG_JOINT_COUNT) continue;
            const Matrix *m=&skin[j];
            float x=capybara_rig_positions[i*3], y=capybara_rig_positions[i*3+1], z=capybara_rig_positions[i*3+2];
            px+=w*(m->m[0]*x+m->m[4]*y+m->m[8]*z+m->m[12]);
            py+=w*(m->m[1]*x+m->m[5]*y+m->m[9]*z+m->m[13]);
            pz+=w*(m->m[2]*x+m->m[6]*y+m->m[10]*z+m->m[14]);
        }
        verts[i*8+0]=px; verts[i*8+1]=py; verts[i*8+2]=pz;
        verts[i*8+3]=verts[i*8+4]=verts[i*8+5]=1.0f;
        verts[i*8+6]=capybara_rig_uvs[i*2];
        verts[i*8+7]=capybara_rig_uvs[i*2+1];
    }
    write_gpu(g->capybara_vertices_gpu,verts,CAPYBARA_VERTEX_BYTES);
}

/* Desenha o modelo 3D e aplica a textura da capivara. */
static void draw_capybara(
    Game *g,
    float x, float foot_y, float z, float yaw,
    float aspect, float camera_x, float camera_z)
{
    Matrix m = model_matrix(
        x, foot_y, z,
        1.0f, 1.0f, 1.0f,
        yaw, aspect, camera_x, camera_z);

    xsdk_gpu_set_texture(
        CUBE_FRAG_SAMPLER_0_0,
        &g->capybara_texture,
        &g->capybara_sampler);

    xsdk_gpu_set_vertices(
        0,
        xsdk_gpu_address(g->capybara_vertices_gpu),
        CAPYBARA_VERTEX_BYTES);

    xsdk_gpu_set_vertex_constants(
        CUBE_VERT_PUSH,
        m.m,
        CUBE_VERT_PUSH_REGISTERS);

    xsdk_gpu_draw_indexed(
        XSDK_PRIMITIVE_TRIANGLE_LIST,
        xsdk_gpu_address(g->capybara_indices_gpu),
        CAPYBARA_INDEX_COUNT,
        false);
}

static bool on_platform(float x, float z)
{
    return x >= PLATFORM_MIN_X &&
           x <= PLATFORM_MAX_X &&
           z >= PLATFORM_MIN_Z &&
           z <= PLATFORM_MAX_Z;
}

static bool collides_obstacles(
    float x, float foot_y, float z)
{
    for (unsigned i = 0; i < OBSTACLE_COUNT; i++) {
        const Obstacle *o = &obstacles[i];

        float min_x =
            o->x - o->sx * 0.5f - PLAYER_RADIUS;
        float max_x =
            o->x + o->sx * 0.5f + PLAYER_RADIUS;

        float min_z =
            o->z - o->sz * 0.5f - PLAYER_RADIUS;
        float max_z =
            o->z + o->sz * 0.5f + PLAYER_RADIUS;

        float bottom = o->y - o->sy * 0.5f;
        float top = o->y + o->sy * 0.5f;

        bool horizontal =
            x > min_x && x < max_x &&
            z > min_z && z < max_z;

        bool vertical =
            foot_y < top - 0.001f &&
            foot_y + PLAYER_HEIGHT > bottom;

        if (horizontal && vertical)
            return true;
    }

    return false;
}

static bool overlaps_obstacle_top(
    const Obstacle *o, float x, float z)
{
    float min_x =
        o->x - o->sx * 0.5f - PLAYER_RADIUS;
    float max_x =
        o->x + o->sx * 0.5f + PLAYER_RADIUS;

    float min_z =
        o->z - o->sz * 0.5f - PLAYER_RADIUS;
    float max_z =
        o->z + o->sz * 0.5f + PLAYER_RADIUS;

    return x >= min_x && x <= max_x &&
           z >= min_z && z <= max_z;
}

static bool land_on_obstacle(
    float x, float z,
    float previous_y, float *foot_y, float *vy)
{
    if (*vy > 0.0f)
        return false;

    for (unsigned i = 0; i < OBSTACLE_COUNT; i++) {
        const Obstacle *o = &obstacles[i];
        float top = o->y + o->sy * 0.5f;

        if (!overlaps_obstacle_top(o, x, z))
            continue;

        if (previous_y >= top && *foot_y <= top) {
            *foot_y = top;
            *vy = 0.0f;
            return true;
        }
    }

    return false;
}

/* Mantida para não alterar as mecânicas já existentes. */

static void draw_checkpoint(
    Game *g, float x, float z,
    float aspect, float camera_x, float camera_z)
{
    Matrix marker = model_matrix(
        x, FLOOR_Y + 0.025f, z,
        0.55f, 0.05f, 0.55f,
        0.0f, aspect, camera_x, camera_z);

    draw_object(g, &marker);
}

int main(void)
{
    xsdk_log_write("GG Fall: capybara model, camera and dash");

    NTSTATUS s = xsdk_splash_show(1500);
    (void)s;

    s = xsdk_gpu_open();
    if (!NT_SUCCESS(s))
        return 1;

    static Game g = {0};

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

    float x = 0.0f;
    float foot_y = FLOOR_Y;
    float z = 0.0f;
    float vy = 0.0f;
    float yaw = 0.0f;
    float walk_phase = 0.0f;
    const float animation_dt = 1.0f / 60.0f;

    float dash_vx = 0.0f;
    float dash_vz = 0.0f;
    float slide_vx = 0.0f;
    float slide_vz = 0.0f;

    int dash_timer = 0;
    int slide_timer = 0;

    bool grounded = true;
    bool has_dashed = false;
    bool finished = false;
    bool previous_a = false;

    float checkpoint_x = 0.0f;
    float checkpoint_z = 0.0f;
    unsigned next_checkpoint = 0;

    for (;;) {
        INPUT_STATE in = {0};

        if (XamInputGetState(0, 0, &in) == ERROR_SUCCESS &&
            (in.gamepad.buttons & INPUT_GAMEPAD_BACK)) {
            break;
        }

        bool a_down =
            (in.gamepad.buttons & INPUT_GAMEPAD_A) != 0;

        bool a_pressed = a_down && !previous_a;
        previous_a = a_down;

        /* Analógico direito controla a câmera orbital. */
        float look_x = axis(in.gamepad.thumb_rx);
        float look_y = axis(in.gamepad.thumb_ry);

        camera_yaw -= look_x * CAMERA_TURN_SPEED;
        camera_pitch -= look_y * CAMERA_PITCH_SPEED;

        if (camera_pitch < CAMERA_MIN_PITCH)
            camera_pitch = CAMERA_MIN_PITCH;

        if (camera_pitch > CAMERA_MAX_PITCH)
            camera_pitch = CAMERA_MAX_PITCH;

        /* Movimento relativo à direção da câmera. */
        float stick_x = axis(in.gamepad.thumb_lx);
        float stick_z = -axis(in.gamepad.thumb_ly);

        float cy = cosf(camera_yaw);
        float sy = sinf(camera_yaw);

        float move_x = stick_x * cy + stick_z * sy;
        float move_z = -stick_x * sy + stick_z * cy;

        bool moving =
            stick_x != 0.0f || stick_z != 0.0f;

        if (moving)
            walk_phase += 0.22f;

        if (!finished) {
            if (dash_timer > 0) {
                float nx = x + dash_vx;
                float nz = z + dash_vz;

                if (!collides_obstacles(nx, foot_y, z))
                    x = nx;

                if (!collides_obstacles(x, foot_y, nz))
                    z = nz;

                dash_timer--;
            } else if (slide_timer > 0) {
                float nx = x + slide_vx;
                float nz = z + slide_vz;

                if (!collides_obstacles(nx, foot_y, z))
                    x = nx;

                if (!collides_obstacles(x, foot_y, nz))
                    z = nz;

                slide_vx *= SLIDE_FRICTION;
                slide_vz *= SLIDE_FRICTION;
                slide_timer--;

                float slide_speed = sqrtf(
                    slide_vx * slide_vx +
                    slide_vz * slide_vz);

                if (slide_speed < SLIDE_STOP_SPEED) {
                    slide_timer = 0;
                    slide_vx = 0.0f;
                    slide_vz = 0.0f;
                }
            } else {
                float candidate_x = x + move_x * SPEED;

                if (!collides_obstacles(
                        candidate_x, foot_y, z)) {
                    x = candidate_x;
                }

                float candidate_z = z + move_z * SPEED;

                if (!collides_obstacles(
                        x, foot_y, candidate_z)) {
                    z = candidate_z;
                }
            }
        }

        if (moving && dash_timer == 0 && slide_timer == 0)
            yaw = atan2f(move_x, move_z);

        /* Pulo e dash no ar. */
        if (a_pressed && grounded && !finished) {
            vy = JUMP_SPEED;
            grounded = false;
            has_dashed = false;
        } else if (a_pressed && !grounded &&
                   !has_dashed && !finished) {
            has_dashed = true;
            dash_timer = DASH_DURATION;

            float forward_x = sinf(yaw);
            float forward_z = cosf(yaw);

            if (moving) {
                float len = sqrtf(
                    move_x * move_x +
                    move_z * move_z);

                if (len > 0.001f) {
                    forward_x = move_x / len;
                    forward_z = move_z / len;
                }
            } else {
                forward_x = sinf(camera_yaw);
                forward_z = -cosf(camera_yaw);
            }

            dash_vx = forward_x * DASH_SPEED;
            dash_vz = forward_z * DASH_SPEED;
            vy = DASH_UP_SPEED;
        }

        float previous_y = foot_y;
        foot_y += vy;

        if (!grounded) {
            if (dash_timer > 0)
                vy -= DASH_GRAVITY;
            else
                vy -= GRAVITY;
        }

        bool landed = false;

        if (vy <= 0.0f &&
            on_platform(x, z) &&
            previous_y >= FLOOR_Y &&
            foot_y <= FLOOR_Y) {
            foot_y = FLOOR_Y;
            vy = 0.0f;
            grounded = true;
            landed = true;
        }

        if (!landed &&
            land_on_obstacle(
                x, z, previous_y, &foot_y, &vy)) {
            grounded = true;
            landed = true;
        }

        /* Ao pousar durante o dash, começa a deslizar. */
        if (landed && dash_timer > 0) {
            slide_timer = SLIDE_DURATION;
            slide_vx = dash_vx;
            slide_vz = dash_vz;
            dash_timer = 0;
        }

        if (grounded && slide_timer == 0)
            has_dashed = false;

        /* Checkpoints em sequência. */
        if (next_checkpoint < CHECKPOINT_COUNT) {
            const Checkpoint *cp =
                &checkpoints[next_checkpoint];

            float dx = x - cp->x;
            float dz = z - cp->z;

            if (dx * dx + dz * dz <=
                CHECKPOINT_RADIUS * CHECKPOINT_RADIUS) {
                checkpoint_x = cp->x;
                checkpoint_z = cp->z;
                next_checkpoint++;

                xsdk_log_write(
                    "GG Fall: checkpoint activated");
            }
        }

        /* Respawn se cair da plataforma. */
        if (foot_y < FALL_LIMIT) {
            x = checkpoint_x;
            z = checkpoint_z;
            foot_y = FLOOR_Y;
            vy = 0.0f;
            grounded = true;

            dash_timer = 0;
            slide_timer = 0;
            has_dashed = false;

            dash_vx = dash_vz = 0.0f;
            slide_vx = slide_vz = 0.0f;
        }

        if (!finished && z <= FINISH_Z) {
            finished = true;
            xsdk_log_write("GG Fall: level completed!");
        }

        camera_y = foot_y + CAMERA_TARGET_HEIGHT;

        uint32_t address;

        s = xsdk_present_begin(
            &address, BEGIN_TIMEOUT_MS);

        if (!NT_SUCCESS(s))
            break;

        float aspect = (float)w / (float)h;

        float camera_x = x * CAMERA_FOLLOW;
        float camera_z = z * CAMERA_FOLLOW;

        xsdk_gpu_set_target(w, h);
        xsdk_gpu_set_depth_stencil(&g.depth);
        xsdk_gpu_set_shaders(&g.vertex, &g.pixel);

        xsdk_built_shader_set_literals(&cube_vert);
        xsdk_built_shader_set_literals(&cube_frag);

        /* Textura e mesh dos objetos do cenário. */
        xsdk_gpu_set_texture(
            CUBE_FRAG_SAMPLER_0_0,
            &g.texture, &g.sampler);

        xsdk_gpu_set_vertices(
            0,
            xsdk_gpu_address(g.vertices),
            VERTICES_BYTES);

        Matrix platform = model_matrix(
            0.0f, -0.94f, -3.5f,
            5.0f, 0.12f, 11.0f,
            0.0f, aspect, camera_x, camera_z);

        draw_object(&g, &platform);

        for (unsigned i = 0; i < CHECKPOINT_COUNT; i++) {
            draw_checkpoint(
                &g, checkpoints[i].x, checkpoints[i].z,
                aspect, camera_x, camera_z);
        }

        for (unsigned i = 0; i < OBSTACLE_COUNT; i++) {
            const Obstacle *o = &obstacles[i];

            Matrix m = model_matrix(
                o->x, o->y, o->z,
                o->sx, o->sy, o->sz,
                o->yaw, aspect, camera_x, camera_z);

            draw_object(&g, &m);
        }

        Matrix finish_line = model_matrix(
            0.0f, FLOOR_Y + 0.025f, FINISH_Z,
            4.5f, 0.05f, 0.18f,
            0.0f, aspect, camera_x, camera_z);

        draw_object(&g, &finish_line);

        /* Atualiza o esqueleto: Run ao mover, Idle quando parado. */
        animate_capybara(&g, moving && grounded && dash_timer == 0 && slide_timer == 0, animation_dt);

        /* Agora desenha a capivara no lugar do boneco de cubos. */
        draw_capybara(
            &g, x, foot_y, z, yaw,
            aspect, camera_x, camera_z);

        xsdk_gpu_resolve(
            address, w, h, CLEAR_ARGB);

        xsdk_present_end();
    }

    xsdk_present_close();

    xsdk_texture_free(&g.capybara_texture);
    xsdk_texture_free(&g.texture);

    xsdk_gpu_free(g.capybara_indices_gpu);
    xsdk_gpu_free(g.capybara_vertices_gpu);

    xsdk_gpu_free(g.indices);
    xsdk_gpu_free(g.vertices);

    xsdk_shader_free(&g.pixel);
    xsdk_shader_free(&g.vertex);

    xsdk_gpu_close();

    return 0;
}
