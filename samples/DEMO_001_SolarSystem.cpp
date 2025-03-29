/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <filesystem>

#include "VulkanApp.h"

#include "ldrutils/lmath/GeometryShapes.h"
#include <ldrutils/lutils/ScopeExit.h>

#include <stb_image.h>

struct Planet {
  float radius;
  float orbitalRadius;
  float globalOrbitalSpeed;
  float localOrbitalSpeed;
  float axialTilt;
  float orbitalInclination;
  const char* textureName;
};

bool g_Paused = false;
bool g_DrawPlanetOrbits = true;

// overall scale of the planetary system
const float g_Scale = 0.001f;

const std::vector<Planet> planets = {
    {1.5f * g_Scale * 110.f, g_Scale * 000, 0.000f, 0.000f, 0.00f, 0.00f, "2k_sun.jpg"},
    {1.5f * g_Scale * 34.5f, g_Scale * 250, 0.650f, 0.650f, 0.03f, 7.01f, "2k_mercury.jpg"},
    {1.5f * g_Scale * 54.3f, g_Scale * 420, 1.969f, 1.969f, 2.64f, 3.39f, "2k_venus_surface.jpg"},
    {1.5f * g_Scale * 44.5f, g_Scale * 600, 0.881f, 0.881f, 23.44f, 0.00f, "2k_earth_daymap.jpg"},
    {1.5f * g_Scale * 36.5f, g_Scale * 800, 1.543f, 1.543f, 25.19f, 1.85f, "2k_mars.jpg"},
    {1.5f * g_Scale * 67.3f, g_Scale * 1350, 2.978f, 2.978f, 3.13f, 1.31f, "2k_jupiter.jpg"},
    {1.5f * g_Scale * 56.1f, g_Scale * 1650, 0.800f, 0.800f, 26.73f, 2.49f, "2k_saturn.jpg"},
    {1.5f * g_Scale * 55.3f, g_Scale * 2000, 0.607f, 0.607f, 82.23f, 0.77f, "2k_uranus.jpg"},
    {1.5f * g_Scale * 54.4f, g_Scale * 2200, 1.700f, 1.700f, 28.32f, 1.77f, "2k_neptune.jpg"},
    {2.0f * g_Scale * 12.5f, g_Scale * 100.0f, 2.843f, 2.843f * 10.0f, 6.68f, 0.0f, "2k_moon.jpg"},
};

enum { Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Moon, TotalPlanets };

const char* codeDefaultVS = R"(
layout (location=0) in vec4 in_Vertex;
layout (location=1) in vec2 in_TexCoord;
layout (location=2) in vec3 in_Normal;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer ModelMatrices {
  mat4 m[];
};

layout(push_constant) uniform constants {
  ModelMatrices bufModelMatrices;
  PerFrame      bufPerFrame;
};

layout (location=0) out vec2 v_TexCoord;
layout (location=1) out vec3 v_WorldPos;
layout (location=2) out vec3 v_WorldNormal;
layout (location=3) out vec3 v_CameraPos;
layout (location=4) flat out uint v_MaterialIndex;

void main() {
  mat4 model = bufModelMatrices.m[gl_BaseInstance];

  v_WorldPos = (model * in_Vertex).xyz;
  v_WorldNormal = transpose(inverse(mat3(model))) * in_Normal;

  gl_Position = bufPerFrame.proj * bufPerFrame.view * model * in_Vertex;

  v_TexCoord  = in_TexCoord;
  v_CameraPos = (inverse(bufPerFrame.view) * vec4( 0.0, 0.0, 0.0, 1.0 )).xyz;

  v_MaterialIndex = gl_BaseInstance;
}
)";

const char* codeDefaultFS = R"(
layout (location=0) in vec2 v_TexCoord;
layout (location=1) in vec3 v_WorldPos;
layout (location=2) in vec3 v_WorldNormal;
layout (location=3) in vec3 v_CameraPos;
layout (location=4) flat in uint v_MaterialIndex;

struct Material {
  vec4 emissive;
  vec4 diffuse;
  uint texEmissive;
  uint texDiffuse;
  uint padding[2];
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material m[];
};

layout(push_constant) uniform constants {
  vec2 bufModelMatrices;
  vec2 bufPerFrame;
  Materials bufMaterials;
};

layout (location=0) out vec4 out_FragColor;

float getDiffuseFactor(vec3 toLight, vec3 normal) {
  float d = dot(toLight, normal);

  return clamp(d, 0.0, 1.0);
}

float pointLight(vec3 lightPos, float lightRadius) {
  vec3 toLight = lightPos - v_WorldPos.xyz;

  float distanceToLight = length(toLight);
  // inspired by https://lisyarus.github.io/blog/posts/point-light-attenuation.html
  float s = distanceToLight / lightRadius;
  float attenuation = max(1.0 - s*s, 0.0);

  return attenuation * getDiffuseFactor(normalize(toLight), normalize(v_WorldNormal));
}

void main() {
  Material m = bufMaterials.m[v_MaterialIndex];

  vec4 Ke = m.emissive * textureBindless2D(m.texEmissive, 0, v_TexCoord);
  vec4 Kd = m.diffuse  * textureBindless2D(m.texDiffuse,  0, v_TexCoord);

  const vec3  lightPos    = vec3(0.0);
  const vec4  lightColor  = vec4(1.0);
  const float lightRadius = 10.0;

  out_FragColor = Ke + Kd * lightColor * pointLight(lightPos, lightRadius);
  out_FragColor.a = Ke.a;
}
)";

const char* codeOrbitVS = R"(
layout (location=0) in vec4 in_Vertex;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer ModelMatrices {
  mat4 m[];
};

layout(push_constant) uniform constants {
  ModelMatrices bufModelMatrices;
  PerFrame bufPerFrame;
  vec2 padding;
};

void main() {
  gl_Position = bufPerFrame.proj * bufPerFrame.view * bufModelMatrices.m[gl_BaseInstance] * in_Vertex;
}
)";

const char* codeOrbitFS = R"(
layout (location=0) out vec4 out_FragColor;

void main() {
  out_FragColor = vec4(1.0, 1.0, 1.0, 0.2);
}
)";

struct SceneNode final {
  SceneNode* parent = nullptr;
  mat4 local = mat4(1.0f);
  mat4 global = mat4(1.0f);
  int materialIdx = -1;
  std::vector<std::shared_ptr<SceneNode>> childNodes_;

  int getMaterialIndexOrParent() const {
    if (materialIdx >= 0)
      return materialIdx;
    assert(parent);
    return parent ? parent->getMaterialIndexOrParent() : -1;
  }

  SceneNode* createNode(const mat4& m = mat4(1)) {
    childNodes_.push_back(std::make_shared<SceneNode>(SceneNode{this, m}));
    return childNodes_.back().get();
  }

  void updateGlobalFromLocal(const mat4& parentTransform) {
    global = parentTransform * local;

    for (auto& i : childNodes_)
      i->updateGlobalFromLocal(global);
  }
};

struct OrbitAnimator final {
  OrbitAnimator(const vec3& axis, const float angle, const float speed, float radius = 0.0f) :
    rotationAxis(axis),
    normalizedRotationAxis(glm::normalize(axis)),
    orbitalRadius(radius),
    rotationSpeed(speed),
    rotationAngle(angle) {}

  void update(float deltaSeconds) {
    rotationAngle = fmodf(rotationAngle + rotationSpeed * deltaSeconds, 360.0f);
    transform = glm::rotate(mat4(1.0f), glm::radians(rotationAngle), normalizedRotationAxis);
  }

  const vec3 rotationAxis = vec3(0.0f);
  const vec3 normalizedRotationAxis = glm::normalize(vec3(1.0f));
  const float orbitalRadius = 0.0f;
  const float rotationSpeed = 0.0f;
  float rotationAngle = 0.0f;
  mat4 transform = mat4(1.0f);
};

struct OrbitAnimationGroup final {
  SceneNode* targetNode = nullptr;
  std::vector<OrbitAnimator> animationGroup_;

  void update(float deltaSeconds) {
    mat4 transform = mat4(1.0f);

    for (OrbitAnimator& anim : animationGroup_) {
      anim.update(deltaSeconds);
      const bool applyTranslation = (anim.orbitalRadius != 0.0f) ? true : false;
      const mat4 translation = applyTranslation ? glm::translate(mat4(1.0f), vec3(0.0f, anim.orbitalRadius, 0.0f)) : mat4(1.0f);
      transform = translation * anim.transform * transform;
    }

    targetNode->local = transform;
  }
};

// shared mesh
struct Mesh final {
  std::vector<GeometryShapes::Vertex> vertices;
  int firstVertex = -1; // the first vertex in the global vertex buffer
};

// unique mesh instance
struct MeshComponent final {
  SceneNode* sceneNode = nullptr;
  std::shared_ptr<Mesh> mesh;
};

struct PerFrameBuffer final {
  mat4 proj = mat4(1.0f);
  mat4 view = mat4(1.0f);
};

struct Material final {
  vec4 emissive = vec4(0.2f, 0.2f, 0.2f, 1.0f);
  vec4 diffuse = vec4(0.8f, 0.8f, 0.8f, 1.0f);
  lvk::TextureHandle texEmissive;
  lvk::TextureHandle texDiffuse;
  lvk::RenderPipelineHandle pipeline;
};

struct Scene final {
  Scene() = default;
  Scene(Scene&) = delete;
  Scene(Scene&&) = default;
  Scene& operator=(Scene&) = delete;
  Scene& operator=(Scene&&) = default;

  std::vector<MeshComponent> meshes;
  std::vector<Material> materials;
  std::vector<OrbitAnimationGroup> animators;
  SceneNode root = SceneNode{nullptr, mat4(1.0f)};

  void updateAnimations(float deltaSeconds) {
    for (OrbitAnimationGroup& animator : animators) {
      animator.update(deltaSeconds);
    }
  }

  void createMesh(SceneNode* node, const std::shared_ptr<Mesh> mesh) {
    meshes.push_back(MeshComponent{node, mesh});
  }

  void createMaterial(SceneNode* node, const Material& mat) {
    materials.emplace_back(mat);
    node->materialIdx = (int)materials.size() - 1;
  }
};

struct ShaderModules final {
  lvk::ShaderModuleHandle vert;
  lvk::ShaderModuleHandle frag;
};

struct VulkanState final {
  std::unordered_map<std::string, lvk::Holder<lvk::TextureHandle>> textures;
  std::vector<lvk::Holder<lvk::ShaderModuleHandle>> shaderModules;
  lvk::Holder<lvk::BufferHandle> bufPerFrame;
  lvk::Holder<lvk::BufferHandle> bufModelMatrices;
  lvk::Holder<lvk::BufferHandle> bufMaterials;
  lvk::Holder<lvk::BufferHandle> bufVertices; // one large vertex buffer for everything
  lvk::Holder<lvk::RenderPipelineHandle> materialDefault;
  lvk::Holder<lvk::RenderPipelineHandle> materialOrbit;
} vulkanState;

lvk::TextureHandle loadTextureFromFile(VulkanApp& app, const std::string& fileName) {
  auto it = vulkanState.textures.find(fileName);

  if (it != vulkanState.textures.end()) {
    return it->second;
  }

  const std::string name = (std::filesystem::path(app.folderContentRoot_) / "src/solarsystem" / fileName).string();

  int w = 0;
  int h = 0;
  int numComponents = 0;

  stbi_set_flip_vertically_on_load(0);
  void* pixels = stbi_load(name.c_str(), &w, &h, &numComponents, 4);

  if (!pixels) {
    LLOGL("Failed to load texture `%s`\n", name.c_str());
    assert(pixels);
    return {};
  }

  SCOPE_EXIT {
    stbi_image_free(pixels);
  };

  lvk::Holder<lvk::TextureHandle> tex = app.ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_RGBA_UN8,
      .dimensions = {uint32_t(w), uint32_t(h)},
      .usage = lvk::TextureUsageBits_Sampled,
      .data = pixels,
      .debugName = fileName.c_str(),
  });

  lvk::TextureHandle handle = tex;

  // ownership
  vulkanState.textures[fileName] = std::move(tex);

  return handle;
}

ShaderModules loadShaderProgram(lvk::IContext* ctx, const char* codeVS, const char* codeFS) {
  lvk::Holder<lvk::ShaderModuleHandle> vert = ctx->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: vert"});
  lvk::Holder<lvk::ShaderModuleHandle> frag = ctx->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: frag"});

  const ShaderModules sm = {vert, frag};

  // ownership
  vulkanState.shaderModules.emplace_back(std::move(vert));
  vulkanState.shaderModules.emplace_back(std::move(frag));

  return sm;
}

Scene createSolarSystemScene(VulkanApp& app) {
  const vec3 X = vec3(1.0f, 0.0f, 0.0f);
  const vec3 Z = vec3(0.0f, 0.0f, 1.0f);

  lvk::IContext* ctx = app.ctx_.get();

#if !defined(ANDROID)
  app.addKeyCallback([](GLFWwindow* window, int key, int, int action, int) {
    if (key == GLFW_KEY_P && action == GLFW_PRESS) {
      g_Paused = !g_Paused;
    }
  });
#endif // !ANDROID

  Scene scene;

  SceneNode* allPlanets[TotalPlanets];

  // attach the Sun to the root node
  allPlanets[Sun] = scene.root.createNode();

  // create all planets
  for (size_t i = 0; i < planets.size(); i++) {
    const Material planetMaterial = {
        .emissive = vec4(.5f, .5f, .5f, 1.0f),
        .diffuse = vec4(0.8f),
        .texEmissive = loadTextureFromFile(app, planets[i].textureName),
        .texDiffuse = loadTextureFromFile(app, planets[i].textureName),
        .pipeline = vulkanState.materialDefault,
    };
    allPlanets[i] = allPlanets[Sun]->createNode(glm::translate(mat4(1.0f), vec3(0.0f, planets[i].orbitalRadius, 0.0f)));
    scene.animators.push_back(OrbitAnimationGroup{allPlanets[i],
                                                  std::vector<OrbitAnimator>{
                                                      {Z, 0.0f, planets[i].localOrbitalSpeed, 0.0f},
                                                      {X, planets[i].axialTilt, 0.0f, planets[i].orbitalRadius},
                                                      {Z, 0.0f, planets[i].globalOrbitalSpeed, 0.0f},
                                                      {X, planets[i].orbitalInclination, 0.0f, 0.0f},
                                                  }});
    scene.createMaterial(allPlanets[i], planetMaterial);
    scene.createMesh(allPlanets[i], std::make_shared<Mesh>(Mesh{GeometryShapes::createIcoSphere(vec3(0), planets[i].radius, 3)}));
  }

  // create orbits
  if (g_DrawPlanetOrbits) {
    const Material orbitMaterial = {
        .pipeline = vulkanState.materialOrbit,
    };
    for (size_t i = 0; i < planets.size(); i++) {
      SceneNode* orbit = allPlanets[Sun]->createNode(glm::rotate(mat4(1.0f), glm::radians(planets[i].orbitalInclination), X));
      scene.createMaterial(orbit, orbitMaterial);
      scene.createMesh(orbit, std::make_shared<Mesh>(Mesh{GeometryShapes::createOrbit(planets[i].orbitalRadius, 128)}));
    }

    // create orbit for the Moon
    SceneNode* orbit = allPlanets[Earth]->createNode();
    scene.createMaterial(orbit, orbitMaterial);
    scene.createMesh(orbit, std::make_shared<Mesh>(Mesh{GeometryShapes::createOrbit(planets[Moon].orbitalRadius, 64)}));
  }

  // adjust initial position
  scene.updateAnimations(150.0);

  return scene;
}

struct RenderOp final {
  lvk::RenderPipelineHandle pipeline;
  uint32_t firstVertex = 0;
  uint32_t numVertices = 0;
  uint32_t materialIndex = 0;
};

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = -80,
      .height = -80,
      .initialCameraPos = vec3(0.8, -1.6, 0.6),
      .initialCameraTarget = vec3(0, 0, 0),
      .initialCameraUpVector = vec3(0, 0, 1),
  };
  VULKAN_APP_DECLARE(app, cfg);

  app.positioner_.maxSpeed_ = 0.1f;

  lvk::IContext* ctx = app.ctx_.get();

  ShaderModules smDefault = loadShaderProgram(ctx, codeDefaultVS, codeDefaultFS);
  ShaderModules smOrbit = loadShaderProgram(ctx, codeOrbitVS, codeOrbitFS);

  vulkanState.materialDefault = ctx->createRenderPipeline({
      .vertexInput =
          {
              .attributes = {{.location = 0, .format = lvk::VertexFormat::Float3, .offset = offsetof(GeometryShapes::Vertex, pos)},
                             {.location = 1, .format = lvk::VertexFormat::Float2, .offset = offsetof(GeometryShapes::Vertex, uv)},
                             {.location = 2, .format = lvk::VertexFormat::Float3, .offset = offsetof(GeometryShapes::Vertex, normal)}},
              .inputBindings = {{.stride = sizeof(GeometryShapes::Vertex)}},
          },
      .smVert = smDefault.vert,
      .smFrag = smDefault.frag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = app.getDepthFormat(),
      .cullMode = lvk::CullMode_None,
      .frontFaceWinding = lvk::WindingMode_CW,
      .debugName = "Pipeline: default",
  });
  vulkanState.materialOrbit = ctx->createRenderPipeline({
      .topology = lvk::Topology_LineStrip,
      .vertexInput =
          {
              .attributes = {{.location = 0, .format = lvk::VertexFormat::Float3, .offset = offsetof(GeometryShapes::Vertex, pos)}},
              .inputBindings = {{.stride = sizeof(GeometryShapes::Vertex)}},
          },
      .smVert = smOrbit.vert,
      .smFrag = smOrbit.frag,
      .color = {{.format = ctx->getSwapchainFormat(),
                 .blendEnabled = true,
                 .srcRGBBlendFactor = lvk::BlendFactor_SrcAlpha,
                 .dstRGBBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha}},
      .depthFormat = app.getDepthFormat(),
      .cullMode = lvk::CullMode_None,
      .frontFaceWinding = lvk::WindingMode_CW,
      .debugName = "Pipeline: orbit",
  });

  vulkanState.bufPerFrame = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Uniform,
      .storage = lvk::StorageType_Device,
      .size = sizeof(PerFrameBuffer),
      .debugName = "Buffer: bufPerFrame",
  });

  Scene scene = createSolarSystemScene(app);

  vulkanState.bufModelMatrices = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = sizeof(mat4) * scene.meshes.size(),
      .debugName = "Buffer: bufModelMatrices",
  });
  // all materials are static - upload them once
  {
    struct MaterialBuffer {
      vec4 emissive;
      vec4 diffuse;
      uint32_t texEmissive;
      uint32_t texDiffuse;
      uint32_t padding[2] = {};
    };
    std::vector<MaterialBuffer> materials;
    for (const Material& m : scene.materials) {
      materials.push_back({
          .emissive = m.emissive,
          .diffuse = m.diffuse,
          .texEmissive = m.texEmissive.index(),
          .texDiffuse = m.texDiffuse.index(),
      });
    }
    vulkanState.bufMaterials = ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = sizeof(MaterialBuffer) * materials.size(),
        .data = materials.data(),
        .debugName = "Buffer: bufMaterials",
    });
  }

  std::vector<RenderOp> flatRenderQueue;
  std::vector<mat4> modelMatrices;

  flatRenderQueue.reserve(scene.meshes.size());
  modelMatrices.reserve(scene.meshes.size());

  std::vector<GeometryShapes::Vertex> allVertices;

  // collect all render ops - the structure of our scene is immutable
  for (MeshComponent& mesh : scene.meshes) {
    assert(mesh.mesh);
    const int materialIdx = mesh.sceneNode->getMaterialIndexOrParent();
    assert(materialIdx >= 0);

    if (mesh.mesh->firstVertex == -1) {
      // pack all different mesh objects into one giant vertex buffer
      mesh.mesh->firstVertex = (int)allVertices.size();
      allVertices.reserve(allVertices.size() + mesh.mesh->vertices.size());
      allVertices.insert(allVertices.end(), mesh.mesh->vertices.begin(), mesh.mesh->vertices.end());
    }

    flatRenderQueue.push_back(RenderOp{
        .pipeline = scene.materials[materialIdx].pipeline,
        .firstVertex = (uint32_t)mesh.mesh->firstVertex,
        .numVertices = (uint32_t)mesh.mesh->vertices.size(),
        .materialIndex = (uint32_t)materialIdx,
    });
    modelMatrices.push_back(mesh.sceneNode->global);
  }

  vulkanState.bufVertices = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Vertex,
      .storage = lvk::StorageType_Device,
      .size = sizeof(GeometryShapes::Vertex) * allVertices.size(),
      .data = allVertices.data(),
      .debugName = "Buffer: vertex",
  });

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    scene.updateAnimations(g_Paused ? 0.0 : deltaSeconds);
    scene.root.updateGlobalFromLocal(mat4(1.0f));

    // update model matrices
    for (size_t i = 0; i != scene.meshes.size(); i++) {
      modelMatrices[i] = scene.meshes[i].sceneNode->global;
    }
    ctx->upload(vulkanState.bufModelMatrices, modelMatrices.data(), sizeof(mat4) * modelMatrices.size());

    const mat4 view = app.camera_.getViewMatrix();
    const mat4 proj = glm::perspective(glm::radians(45.0f), aspectRatio, 0.01f, 100.0f);

    lvk::ICommandBuffer& buf = ctx->acquireCommandBuffer();
    const lvk::Framebuffer fb = {
        .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
        .depthStencil = {.texture = app.getDepthTexture()},
    };
    // render
    {
      const struct {
        uint64_t bufModelMatrices;
        uint64_t bufPerFrame;
        uint64_t bufMaterials;
      } pc = {
          .bufModelMatrices = ctx->gpuAddress(vulkanState.bufModelMatrices),
          .bufPerFrame = ctx->gpuAddress(vulkanState.bufPerFrame),
          .bufMaterials = ctx->gpuAddress(vulkanState.bufMaterials),
      };

      buf.cmdBindVertexBuffer(0, vulkanState.bufVertices, 0);

      buf.cmdUpdateBuffer(vulkanState.bufPerFrame,
                          PerFrameBuffer{
                              .proj = proj,
                              .view = view,
                          });
      buf.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
                             .depth = {.loadOp = lvk::LoadOp_Clear, .clearDepth = 1.0f}},
                            fb);

      // all pipelines share the same push constants - bind them up front
      buf.cmdBindRenderPipeline(flatRenderQueue[0].pipeline);
      buf.cmdPushConstants(pc);

      buf.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});
      for (const RenderOp& ROP : flatRenderQueue) {
        buf.cmdBindRenderPipeline(ROP.pipeline);
        buf.cmdDraw(ROP.numVertices, 1, ROP.firstVertex, ROP.materialIndex);
      }

      buf.cmdEndRendering();
    }
    // ImGui pass
    buf.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Load}}, .depth = {.loadOp = lvk::LoadOp_DontCare}}, fb);
    app.imgui_->beginFrame(fb);
    auto imGuiPushFlagsAndStyles = [](bool value) {
      ImGui::BeginDisabled(!value);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * (value ? 1.0f : 0.3f));
    };
    auto imGuiPopFlagsAndStyles = []() {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    };

    const float indentSize = 16.0f;
    ImGui::Begin("Keyboard hints:", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
#if !defined(ANDROID)
    ImGui::Text("W/S/A/D - camera movement");
    ImGui::Text("1/2 - camera up/down");
    ImGui::Text("Shift - fast movement");
    ImGui::Text("Space - reset camera");
    ImGui::Separator();
#endif
    ImGui::Checkbox("Pause animation (P)", &g_Paused);
    ImGui::End();
    app.drawFPS();
    app.imgui_->endFrame(buf);
    buf.cmdEndRendering();
    ctx->submit(buf, ctx->getCurrentSwapchainTexture());
  });

  vulkanState = {};

  VULKAN_APP_EXIT();
}
