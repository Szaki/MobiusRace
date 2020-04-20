#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

int majorVersion = 3, minorVersion = 3;

struct vec2
{
    float x, y;
    vec2(float _x = 0, float _y = 0)
    {
        x = _x;
        y = _y;
    }
};

struct vec3
{
    float x, y, z;
    vec3(float _x = 0, float _y = 0, float _z = 0)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
    vec3 operator+(const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    vec3 operator-(const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    vec3 operator*(const vec3 &v) const { return vec3(x * v.x, y * v.y, z * v.z); }
    vec3 operator/(const float v) const { return vec3(x / v, y / v, z / v); }
    vec3 operator-() const { return vec3(-x, -y, -z); }
    vec3 normalize() const { return (*this) * (1.0f / (Length() + 0.000001)); }
    float Length() const { return sqrtf(x * x + y * y + z * z); }

    void SetUniform(unsigned shaderProg, const char *name)
    {
        int location = glGetUniformLocation(shaderProg, name);
        if (location >= 0)
            glUniform3fv(location, 1, &x);
        else
            printf("uniform %s cannot be set\n", name);
    }
};

float dot(const vec3 &v1, const vec3 &v2) { return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

vec3 cross(const vec3 &v1, const vec3 &v2) { return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }

struct vec4
{
    float x, y, z, w;
    vec4(float _x = 0, float _y = 0, float _z = 0, float _w = 1)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }

    void SetUniform(unsigned shaderProg, const char *name)
    {
        int location = glGetUniformLocation(shaderProg, name);
        if (location >= 0)
            glUniform4fv(location, 1, &x);
        else
            printf("uniform %s cannot be set\n", name);
    }
};

struct mat4
{
    float m[4][4];

public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33)
    {
        m[0][0] = m00;
        m[0][1] = m01;
        m[0][2] = m02;
        m[0][3] = m03;
        m[1][0] = m10;
        m[1][1] = m11;
        m[1][2] = m12;
        m[1][3] = m13;
        m[2][0] = m20;
        m[2][1] = m21;
        m[2][2] = m22;
        m[2][3] = m23;
        m[3][0] = m30;
        m[3][1] = m31;
        m[3][2] = m32;
        m[3][3] = m33;
    }

    mat4 operator*(const mat4 &right)
    {
        mat4 result;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++)
                    result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }

    void SetUniform(unsigned shaderProg, const char *name)
    {
        int location = glGetUniformLocation(shaderProg, name);
        if (location >= 0)
            glUniformMatrix4fv(location, 1, GL_TRUE, &m[0][0]);
        else
            printf("uniform %s cannot be set\n", name);
    }
};

mat4 TranslateMatrix(vec3 t)
{
    return mat4(1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                t.x, t.y, t.z, 1);
}

mat4 ScaleMatrix(vec3 s)
{
    return mat4(s.x, 0, 0, 0,
                0, s.y, 0, 0,
                0, 0, s.z, 0,
                0, 0, 0, 1);
}

mat4 RotationMatrix(float angle, vec3 w)
{
    float c = cosf(angle), s = sinf(angle);
    w = w.normalize();
    return mat4(c * (1 - w.x * w.x) + w.x * w.x, w.x * w.y * (1 - c) + w.z * s, w.x * w.z * (1 - c) - w.y * s, 0,
                w.x * w.y * (1 - c) - w.z * s, c * (1 - w.y * w.y) + w.y * w.y, w.y * w.z * (1 - c) + w.x * s, 0,
                w.x * w.z * (1 - c) + w.y * s, w.y * w.z * (1 - c) - w.x * s, c * (1 - w.z * w.z) + w.z * w.z, 0,
                0, 0, 0, 1);
}

struct Material
{
    vec3 kd, ks, ka;
    float shininess;
};

struct Light
{
    vec3 La, Le;
    vec4 wLightPos;
};

struct Texture
{
    unsigned int textureId;

    Texture(const int width, const int height) { glGenTextures(1, &textureId); }

    void SetUniform(unsigned shaderProg, const char *samplerName, unsigned int textureUnit = 0)
    {
        int location = glGetUniformLocation(shaderProg, samplerName);
        if (location >= 0)
        {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        else
            printf("uniform %s cannot be set\n", samplerName);
    }
};

struct RenderState
{
    mat4 MVP, M, Minv, V, P;
    Material *material;
    Light light;
    Texture *texture;
    vec3 wEye;
};

struct VertexData
{
    vec3 position, normal;
    vec2 texcoord;
};

struct ColorfulTexture : public Texture
{
    ColorfulTexture(const vec3 col1, const vec3 col2, const int width = 0, const int height = 0) : Texture(width, height)
    {
        glBindTexture(GL_TEXTURE_2D, textureId);
        std::vector<vec3> image(width * height);
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
            {
                image[y * width + x] = (x & 1) ^ (y & 1) ? col1 : col2;
            }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
};

class PhongShader
{
    unsigned int shaderProgram;

    const char *vertexSource = R"(
        #version 330
        precision highp float;
 
        uniform mat4  MVP, M, Minv; 
        uniform vec4  wLiPos;       
        uniform vec3  wEye;         
 
        layout(location = 0) in vec3  vtxPos;            
        layout(location = 1) in vec3  vtxNorm;           
        layout(location = 2) in vec2  vtxUV;
 
        out vec3 wNormal;            
        out vec3 wView;             
        out vec3 wLight;            
        out vec2 texcoord;
 
        void main() {
           gl_Position = vec4(vtxPos, 1) * MVP; 
           
           vec4 wPos = vec4(vtxPos, 1) * M;
           wLight = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
           wView  = wEye * wPos.w - wPos.xyz;
           wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
 
           texcoord = vtxUV;
        }
    )";

    const char *fragmentSource = R"(
        #version 330
        precision highp float;
 
        uniform vec3 kd, ks, ka; 
        uniform vec3 La, Le;     
        uniform float shine;     
        uniform sampler2D diffuseTexture;
 
        in  vec3 wNormal;       
        in  vec3 wView;         
        in  vec3 wLight;        
        in vec2 texcoord;
        out vec4 fragmentColor; 
 
        void main() {
            vec3 N = normalize(wNormal);
            vec3 V = normalize(wView); 
            vec3 L = normalize(wLight);
            vec3 H = normalize(L + V);
            float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
            vec3 texColor = texture(diffuseTexture, texcoord).rgb;
 
            
            vec3 color = ka * texColor * La + (kd * texColor * cost + ks * pow(cosd,shine)) * Le;
            fragmentColor = vec4(color, 1);
        }
    )";

public:
    void getErrorInfo(unsigned int handle)
    {
        int logLen, written;
        glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0)
        {
            char *log = new char[logLen];
            glGetShaderInfoLog(handle, logLen, &written, log);
            printf("Shader log:\n%s", log);
            delete log;
        }
    }
    void checkShader(unsigned int shader, const char *message)
    {
        int OK;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
        if (!OK)
        {
            printf("%s!\n", message);
            getErrorInfo(shader);
            getchar();
        }
    }
    void checkLinking(unsigned int program)
    {
        int OK;
        glGetProgramiv(program, GL_LINK_STATUS, &OK);
        if (!OK)
        {
            printf("Failed to link shader program!\n");
            getErrorInfo(program);
            getchar();
        }
    }

    void Create(const char *vertexSource, const char *fragmentSource, const char *fsOuputName)
    {
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        if (!vertexShader)
        {
            printf("Error in vertex shader creation\n");
            exit(1);
        }
        glShaderSource(vertexShader, 1, &vertexSource, NULL);
        glCompileShader(vertexShader);
        checkShader(vertexShader, "Vertex shader error");

        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fragmentShader)
        {
            printf("Error in fragment shader creation\n");
            exit(1);
        }
        glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(fragmentShader);
        checkShader(fragmentShader, "Fragment shader error");

        shaderProgram = glCreateProgram();
        if (!shaderProgram)
        {
            printf("Error in shader program creation\n");
            exit(1);
        }
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);

        glBindFragDataLocation(shaderProgram, 0, fsOuputName);

        glLinkProgram(shaderProgram);
        checkLinking(shaderProgram);
    }

    PhongShader()
    {
        Create(vertexSource, fragmentSource, "fragmentColor");
    }

    void Bind(RenderState state)
    {
        glUseProgram(shaderProgram);
        state.MVP.SetUniform(shaderProgram, "MVP");
        state.M.SetUniform(shaderProgram, "M");
        state.Minv.SetUniform(shaderProgram, "Minv");
        state.wEye.SetUniform(shaderProgram, "wEye");
        state.material->kd.SetUniform(shaderProgram, "kd");
        state.material->ks.SetUniform(shaderProgram, "ks");
        state.material->ka.SetUniform(shaderProgram, "ka");
        int location = glGetUniformLocation(shaderProgram, "shine");
        if (location >= 0)
            glUniform1f(location, state.material->shininess);
        else
            printf("uniform shininess cannot be set\n");
        state.light.La.SetUniform(shaderProgram, "La");
        state.light.Le.SetUniform(shaderProgram, "Le");
        state.light.wLightPos.SetUniform(shaderProgram, "wLiPos");
        state.texture->SetUniform(shaderProgram, "diffuseTexture");
    }

    ~PhongShader()
    {
        glDeleteProgram(shaderProgram);
    }
};

PhongShader *shader;

class Geometry
{
    unsigned int vao, type;

protected:
    int nVertices;

public:
    Geometry(unsigned int _type)
    {
        type = _type;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }
    void Draw()
    {
        glBindVertexArray(vao);
        glDrawArrays(type, 0, nVertices);
    }
};

struct ParamSurface : public Geometry
{
    ParamSurface() : Geometry(GL_TRIANGLES) {}

    virtual VertexData GenVertexData(float u, float v) = 0;

    void Create(int N = 16, int M = 16)
    {
        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        std::vector<VertexData> vtxData;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                vtxData.push_back(GenVertexData((float)i / N, (float)j / M));
                vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
                vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
                vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
                vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)(j + 1) / M));
                vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
            }
        }
        nVertices = vtxData.size();
        glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, texcoord));
    }
};

struct HoleySurface : public Geometry
{
    HoleySurface() : Geometry(GL_TRIANGLES) {}

    virtual VertexData GenVertexData(float u, float v) = 0;

    void Create(int N = 16, int M = 16)
    {
        unsigned int vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        std::vector<VertexData> vtxData;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                if (j % 4 == 1 && i % 4 == 0)
                    continue;
                vtxData.push_back(GenVertexData((float)i / N, (float)j / M));
                vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
                vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
                vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
                vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)(j + 1) / M));
                vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
            }
        }
        nVertices = vtxData.size();
        glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, texcoord));
    }
};

class Sphere : public ParamSurface
{
    float rad;

public:
    Sphere(float r, int t1, int t2)
    {
        rad = r;
        Create(t1, t2);
    }

    VertexData GenVertexData(float u, float v)
    {
        VertexData vd;
        float U = u * 2.0 * M_PI, V = v * M_PI;
        vd.normal = vec3(cos(U) * sin(V), sin(U) * sin(V), cos(V));
        vd.position = vd.normal * rad;
        vd.texcoord = vec2(u, v);
        return vd;
    }
};

class Mobius : public ParamSurface
{
    float R, w;

public:
    Mobius(float _R, float _w, int t1, int t2)
    {
        R = _R;
        w = _w;
        Create(t1, t2);
    }

    VertexData GenVertexData(float u, float v)
    {
        VertexData vd;
        float U = u * M_PI, V = (v - 0.5f) * w;
        float C2U = cos(2 * U), S2U = sin(2 * U);
        float D = R + V * cos(U), dDdU = -V * sin(U), dDdV = cos(U);
        vd.position = vec3(D * C2U, D * S2U, V * sin(U));
        vec3 drdU(dDdU * C2U - D * S2U * 2, dDdU * S2U + D * C2U * 2, V * cos(U));
        vec3 drdV(dDdV * C2U, dDdV * S2U, sin(U));
        vd.normal = cross(drdU, drdV);
        vd.texcoord = vec2(u, v);
        return vd;
    }
};

class HoleyMobius : public HoleySurface
{
    float R, w;

public:
    HoleyMobius(float _R, float _w, int t1, int t2)
    {
        R = _R;
        w = _w;
        Create(t1, t2);
    }

    VertexData GenVertexData(float u, float v)
    {
        VertexData vd;
        float U = u * M_PI, V = (v - 0.5f) * w;
        float C2U = cos(2 * U), S2U = sin(2 * U);
        float D = R + V * cos(U), dDdU = -V * sin(U), dDdV = cos(U);
        vd.position = vec3(D * C2U, D * S2U, V * sin(U));
        vec3 drdU(dDdU * C2U - D * S2U * 2, dDdU * S2U + D * C2U * 2, V * cos(U));
        vec3 drdV(dDdV * C2U, dDdV * S2U, sin(U));
        vd.normal = cross(drdU, drdV);
        vd.texcoord = vec2(u, v);
        return vd;
    }
};

struct MobiusData
{
    float R, w;

    MobiusData(float _R, float _w)
    {
        R = _R;
        w = _w;
    }

    vec3 GetdrdU(float u, float v)
    {
        float U = u * M_PI, V = (v - 0.5f) * w;
        float C2U = cos(2 * U), S2U = sin(2 * U);
        float D = R + V * cos(U), dDdU = -V * sin(U), dDdV = cos(U);
        return vec3(dDdU * C2U - D * S2U * 2, dDdU * S2U + D * C2U * 2, V * cos(U)).normalize();
    }

    vec3 GetdrdV(float u, float v)
    {
        float U = u * M_PI, V = (v - 0.5f) * w;
        float C2U = cos(2 * U), S2U = sin(2 * U);
        float D = R + V * cos(U), dDdU = -V * sin(U), dDdV = cos(U);
        return vec3(dDdV * C2U, dDdV * S2U, sin(U)).normalize();
    }

    vec3 GetNormal(float u, float v)
    {
        return cross(GetdrdU(u, v), GetdrdV(u, v)).normalize();
    }

    vec3 GetPoint(float u, float v)
    {
        float U = u * M_PI, V = (v - 0.5f) * w;
        float C2U = cos(2 * U), S2U = sin(2 * U);
        float D = R + V * cos(U), dDdU = -V * sin(U), dDdV = cos(U);
        return vec3(D * C2U, D * S2U, V * sin(U));
    }
};

class Torus : public ParamSurface
{
    float r, R;

    vec3 Point(float u, float v, float rr)
    {
        float U = u * 2.0 * M_PI, V = v * 2.0 * M_PI;
        float D = R + rr * cos(U);
        return vec3(D * cos(V), D * sin(V), rr * sin(U));
    }

public:
    Torus(float _r, float _R, int t1, int t2)
    {
        r = _r;
        R = _R;
        Create(t1, t2);
    }

    VertexData GenVertexData(float u, float v)
    {
        VertexData vd;
        vd.position = Point(u, v, r);
        vd.normal = vd.position - Point(u, v, 0);
        vd.texcoord = vec2(u, v);
        return vd;
    }
};

struct Object
{
    Material *material;
    Texture *texture;
    Geometry *geometry;
    vec3 translation, rotationAxis;
    float rotationAngle;

public:
    Object(Material *_material, Texture *_texture, Geometry *_geometry, vec3 trans = vec3(0, 0, 0)) : translation(trans), rotationAxis(0, 0, 1), rotationAngle(0)
    {
        texture = _texture;
        material = _material;
        geometry = _geometry;
    }

    virtual void Draw(RenderState state)
    {
        state.M = RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis);
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }

    virtual void Animate(float time)
    {
    }
};

struct RollingObject : public Object
{
    float u, v;
    float rad;
    float start, speed;
    MobiusData *mobdata;

public:
    RollingObject(Material *_material, Texture *_texture, Geometry *_geometry, float _r, float lane, float s, MobiusData *m) : Object(_material, _texture, _geometry)
    {
        v = lane;
        mobdata = m;
        rad = _r;
        speed = s;
    }

    void Draw(RenderState state)
    {
        state.M = RotationMatrix((-u) * M_PI * 2.0f / rad, vec3(0.0f, 0.0f, 1.0f)) *
                  RotationMatrix((u + 0.5f) * M_PI, -mobdata->GetdrdU(u, v)) *
                  TranslateMatrix(mobdata->GetPoint(u, v) + mobdata->GetNormal(u, v) * rad);
        state.Minv = TranslateMatrix(-(mobdata->GetPoint(u, v) + mobdata->GetNormal(u, v) * rad)) *
                     RotationMatrix((-u - 0.5f) * M_PI, -mobdata->GetdrdU(u, v)) *
                     RotationMatrix(u * M_PI * 2.0f / rad, vec3(0.0f, 0.0f, 1.0f));
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }

    void Animate(float time)
    {
        u = -time * speed - 0.5f;
    }
};

struct Camera
{
    vec3 wEye, wLookat, wVup;
    float fov, asp, fp, bp;
    bool moving;
    float speed;
    MobiusData *mobdata;
    float v;

public:
    Camera(MobiusData *m, float s)
    {
        asp = 1;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 0.01f;
        bp = 10.0f;
        mobdata = m;
        speed = s;
        v = 0.5f;
    }
    mat4 V()
    {
        vec3 w = (wEye - wLookat).normalize();
        vec3 u = cross(wVup, w).normalize();
        vec3 v = cross(w, u);
        return TranslateMatrix(-wEye) * mat4(u.x, v.x, w.x, 0,
                                             u.y, v.y, w.y, 0,
                                             u.z, v.z, w.z, 0,
                                             0, 0, 0, 1);
    }
    mat4 P()
    {
        return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
                    0, 1 / tan(fov / 2), 0, 0,
                    0, 0, -(fp + bp) / (bp - fp), -1,
                    0, 0, -2 * fp * bp / (bp - fp), 0);
    }
    void Animate(float time)
    {
        time = time * speed;
        if (moving)
        {
            wEye = mobdata->GetPoint(time, v) + mobdata->GetNormal(time, v) * 0.5;
            wLookat = wEye + mobdata->GetdrdU(time, v) * 0.5;
            wVup = mobdata->GetNormal(time, v);
        }
        else
        {
            wEye = vec3(0, 0, 6);
            wLookat = vec3(0, 0, 0);
            wVup = vec3(0, 1, 0);
        }
    }
};

MobiusData *mobdata = new MobiusData(3, 2);
Camera camera(mobdata, 0.3f);

class Scene
{
    std::vector<Object *> objects;
    Light light;

public:
    void Build()
    {
        shader = new PhongShader();

        Material *black = new Material;

        Material *diff = new Material;
        diff->kd = vec3(1.0f, 0.1f, 0.2f);
        diff->ks = vec3(0.0, 0.0, 0.0);
        diff->ka = vec3(0.2f, 0.2f, 0.2f);
        diff->shininess = 50;

        Material *spec = new Material;
        spec->kd = vec3(0.3f, 0.3f, 0.3f);
        spec->ks = vec3(1, 1, 1);
        spec->ka = vec3(0.1f, 0.1f, 0.1f);
        spec->shininess = 100;

        Material *starmat = new Material;
        starmat->kd = vec3(1.0f, 1.0f, 1.0f);
        starmat->ks = vec3(1, 1, 1);
        starmat->ka = vec3(1.0f, 1.0f, 1.0f);
        starmat->shininess = 50;

        Texture *ob = new ColorfulTexture(vec3(0, 0, 0), vec3(1, 0.5, 0), 4, 8);
        Texture *magenta = new ColorfulTexture(vec3(0, 0, 0), vec3(1, 0, 1), 5, 10);
        Texture *stars = new ColorfulTexture(vec3(1, 1, 1), vec3(1, 1, 1), 1, 1);
        Texture *blueyellow = new ColorfulTexture(vec3(0, 0, 1), vec3(1, 1, 0), 5, 8);
        Texture *bw = new ColorfulTexture(vec3(0, 0, 0), vec3(1, 1, 1), 3, 6);

        Geometry *torus = new Torus(0.1, 0.2, 20, 20);
        Geometry *mobius = new HoleyMobius(3, 2, 30, 30);
        Geometry *sphere = new Sphere(0.3, 30, 30);
        Geometry *skysphere = new Sphere(10, 40, 40);
        Geometry *starsphere = new Sphere(0.04, 4, 4);

        objects.push_back(new Object(black, stars, skysphere));
        objects.push_back(new Object(diff, ob, mobius));
        objects.push_back(new RollingObject(spec, bw, torus, 0.3, 0.0, 0.35, mobdata));
        objects.push_back(new RollingObject(spec, blueyellow, torus, 0.3, 1.0, 0.2, mobdata));
        objects.push_back(new RollingObject(spec, ob, torus, 0.3, 0.66, 0.25, mobdata));
        objects.push_back(new RollingObject(spec, magenta, sphere, 0.3, 0.33, 0.3, mobdata));
        for (int i = 0; i < 150; i++)
        {
            vec3 random(rand() % 20 - 10, rand() % 20 - 10, rand() % 20 - 10);
            random = random.normalize() * 9.0;
            objects.push_back(new Object(starmat, stars, starsphere, random));
        }

        light.wLightPos = vec4(0, 0, 7, 0);
        light.La = vec3(1, 1, 1);
        light.Le = vec3(4, 4, 4);
    }

    void Render()
    {
        RenderState state;
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.light = light;
        for (Object *obj : objects)
            obj->Draw(state);
    }

    void Animate(float time)
    {
        camera.Animate(time);
        for (Object *obj : objects)
            obj->Animate(time);
    }
};

Scene scene;

void onInitialization()
{
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}

void onDisplay()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    scene.Render();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY)
{
    if (key == 32)
    {
        if (camera.moving)
            camera.moving = false;
        else
        {
            camera.moving = true;
        }
    };
    if (key == 'w')
        camera.v += 0.1f;
    if (key == 'p')
        camera.v -= 0.1f;
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void onMouseMotion(int pX, int pY)
{
}

void onIdle()
{
    float time = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    scene.Animate(time);
    glutPostRedisplay();
}

void onExit()
{
    printf("exit");
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(100, 100);
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
    glewExperimental = true;
    glewInit();
#endif

    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    onInitialization();

    glutDisplayFunc(onDisplay);
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    onExit();
    return 1;
}
