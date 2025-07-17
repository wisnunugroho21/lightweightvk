/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include <minilog/minilog.h>

#if defined(ANDROID)
#include <android/native_window.h>
#endif

// clang-format off
#if defined(LVK_WITH_TRACY)
  #include "tracy/Tracy.hpp"
  // predefined RGB colors for "heavy" point-of-interest operations
  #define LVK_PROFILER_COLOR_WAIT 0xff0000
  #define LVK_PROFILER_COLOR_SUBMIT 0x0000ff
  #define LVK_PROFILER_COLOR_PRESENT 0x00ff00
  #define LVK_PROFILER_COLOR_CREATE 0xff6600
  #define LVK_PROFILER_COLOR_DESTROY 0xffa500
  #define LVK_PROFILER_COLOR_BARRIER 0xffffff
  #define LVK_PROFILER_COLOR_CMD_DRAW 0x8b0000
  #define LVK_PROFILER_COLOR_CMD_COPY 0x8b0a50
  #define LVK_PROFILER_COLOR_CMD_RTX 0x8b0000
  #define LVK_PROFILER_COLOR_CMD_DISPATCH 0x8b0000
  //
  #define LVK_PROFILER_FUNCTION() ZoneScoped
  #define LVK_PROFILER_FUNCTION_COLOR(color) ZoneScopedC(color)
  #define LVK_PROFILER_ZONE(name, color) \
    {                                    \
      ZoneScopedC(color);                \
      ZoneName(name, strlen(name))
  #define LVK_PROFILER_ZONE_END() }
  #define LVK_PROFILER_THREAD(name) tracy::SetThreadName(name)
  #define LVK_PROFILER_FRAME(name) FrameMarkNamed(name)
#else
  #define LVK_PROFILER_FUNCTION()
  #define LVK_PROFILER_FUNCTION_COLOR(color)
  #define LVK_PROFILER_ZONE(name, color) {
  #define LVK_PROFILER_ZONE_END() }
  #define LVK_PROFILER_THREAD(name)
  #define LVK_PROFILER_FRAME(name)
#endif // LVK_WITH_TRACY
// clang-format on

#define LVK_ARRAY_NUM_ELEMENTS(x) (sizeof(x) / sizeof((x)[0]))

namespace lvk {

class IContext;

bool Assert(bool cond, const char* file, int line, const char* format, ...);

// Non-ref counted handles; based on:
// https://enginearchitecture.realtimerendering.com/downloads/reac2023_modern_mobile_rendering_at_hypehype.pdf
template<typename ObjectType>
class Handle final {
 public:
  Handle() = default;
  explicit Handle(void* ptr) :
    index_(reinterpret_cast<ptrdiff_t>(ptr) & 0xffffffff), gen_((reinterpret_cast<ptrdiff_t>(ptr) >> 32) & 0xffffffff) {}

  bool empty() const {
    return gen_ == 0;
  }
  bool valid() const {
    return gen_ != 0;
  }
  uint32_t index() const {
    return index_;
  }
  uint32_t gen() const {
    return gen_;
  }
  void* indexAsVoid() const {
    return reinterpret_cast<void*>(static_cast<ptrdiff_t>(index_));
  }
  void* handleAsVoid() const {
    static_assert(sizeof(void*) >= sizeof(uint64_t));
    return reinterpret_cast<void*>((static_cast<ptrdiff_t>(gen_) << 32) + static_cast<ptrdiff_t>(index_));
  }
  bool operator==(const Handle<ObjectType>& other) const {
    return index_ == other.index_ && gen_ == other.gen_;
  }
  bool operator!=(const Handle<ObjectType>& other) const {
    return index_ != other.index_ || gen_ != other.gen_;
  }
  // allow conditions 'if (handle)'
  explicit operator bool() const {
    return gen_ != 0;
  }

 private:
  Handle(uint32_t index, uint32_t gen) : index_(index), gen_(gen){};

  template<typename ObjectType_, typename ImplObjectType>
  friend class Pool;

  uint32_t index_ = 0;
  uint32_t gen_ = 0;
};

static_assert(sizeof(Handle<class Foo>) == sizeof(uint64_t));

// specialized with dummy structs for type safety
using ComputePipelineHandle = lvk::Handle<struct ComputePipeline>;
using RenderPipelineHandle = lvk::Handle<struct RenderPipeline>;
using RayTracingPipelineHandle = lvk::Handle<struct RayTracingPipeline>;
using ShaderModuleHandle = lvk::Handle<struct ShaderModule>;
using SamplerHandle = lvk::Handle<struct Sampler>;
using BufferHandle = lvk::Handle<struct Buffer>;
using TextureHandle = lvk::Handle<struct Texture>;
using QueryPoolHandle = lvk::Handle<struct QueryPool>;
using AccelStructHandle = lvk::Handle<struct AccelerationStructure>;

// forward declarations to access incomplete type IContext
void destroy(lvk::IContext* ctx, lvk::ComputePipelineHandle handle);
void destroy(lvk::IContext* ctx, lvk::RenderPipelineHandle handle);
void destroy(lvk::IContext* ctx, lvk::RayTracingPipelineHandle handle);
void destroy(lvk::IContext* ctx, lvk::ShaderModuleHandle handle);
void destroy(lvk::IContext* ctx, lvk::SamplerHandle handle);
void destroy(lvk::IContext* ctx, lvk::BufferHandle handle);
void destroy(lvk::IContext* ctx, lvk::TextureHandle handle);
void destroy(lvk::IContext* ctx, lvk::QueryPoolHandle handle);
void destroy(lvk::IContext* ctx, lvk::AccelStructHandle handle);

template<typename HandleType>
class Holder final {
 public:
  Holder() = default;
  Holder(lvk::IContext* ctx, HandleType handle) : ctx_(ctx), handle_(handle) {}
  ~Holder() {
    lvk::destroy(ctx_, handle_);
  }
  Holder(const Holder&) = delete;
  Holder(Holder&& other) : ctx_(other.ctx_), handle_(other.handle_) {
    other.ctx_ = nullptr;
    other.handle_ = HandleType{};
  }
  Holder& operator=(const Holder&) = delete;
  Holder& operator=(Holder&& other) {
    std::swap(ctx_, other.ctx_);
    std::swap(handle_, other.handle_);
    return *this;
  }
  Holder& operator=(std::nullptr_t) {
    this->reset();
    return *this;
  }

  inline operator HandleType() const {
    return handle_;
  }

  bool valid() const {
    return handle_.valid();
  }

  bool empty() const {
    return handle_.empty();
  }

  void reset() {
    lvk::destroy(ctx_, handle_);
    ctx_ = nullptr;
    handle_ = HandleType{};
  }

  HandleType release() {
    ctx_ = nullptr;
    return std::exchange(handle_, HandleType{});
  }

  uint32_t gen() const {
    return handle_.gen();
  }
  uint32_t index() const {
    return handle_.index();
  }
  void* indexAsVoid() const {
    return handle_.indexAsVoid();
  }
  void* handleAsVoid() const {
    return handle_.handleAsVoid();
  }

 private:
  lvk::IContext* ctx_ = nullptr;
  HandleType handle_ = {};
};

} // namespace lvk

// clang-format off
#if !defined(NDEBUG) && (defined(DEBUG) || defined(_DEBUG) || defined(__DEBUG))
  #define LVK_VERIFY(cond) ::lvk::Assert((cond), __FILE__, __LINE__, #cond)
  #define LVK_ASSERT(cond) (void)LVK_VERIFY(cond)
  #define LVK_ASSERT_MSG(cond, format, ...) (void)::lvk::Assert((cond), __FILE__, __LINE__, (format), ##__VA_ARGS__)
#else
  #define LVK_VERIFY(cond) (cond)
  #define LVK_ASSERT(cond)
  #define LVK_ASSERT_MSG(cond, format, ...)
#endif
// clang-format on

namespace lvk {

enum { LVK_MAX_COLOR_ATTACHMENTS = 8 };
enum { LVK_MAX_MIP_LEVELS = 16 };
enum { LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE = 4 };

enum IndexFormat : uint8_t {
  IndexFormat_UI8,
  IndexFormat_UI16,
  IndexFormat_UI32,
};

enum Topology : uint8_t {
  Topology_Point,
  Topology_Line,
  Topology_LineStrip,
  Topology_Triangle,
  Topology_TriangleStrip,
  Topology_Patch,
};

enum ColorSpace : uint8_t {
  ColorSpace_SRGB_NONLINEAR,
  ColorSpace_SRGB_EXTENDED_LINEAR,
  ColorSpace_HDR10,
  ColorSpace_BT709_LINEAR,
};

enum TextureType : uint8_t {
  TextureType_2D,
  TextureType_3D,
  TextureType_Cube,
};

enum SamplerFilter : uint8_t { SamplerFilter_Nearest = 0, SamplerFilter_Linear };
enum SamplerMip : uint8_t { SamplerMip_Disabled = 0, SamplerMip_Nearest, SamplerMip_Linear };
enum SamplerWrap : uint8_t { SamplerWrap_Repeat = 0, SamplerWrap_Clamp, SamplerWrap_MirrorRepeat };

enum HWDeviceType {
  HWDeviceType_Discrete = 1,
  HWDeviceType_External = 2,
  HWDeviceType_Integrated = 3,
  HWDeviceType_Software = 4,
};

struct HWDeviceDesc {
  enum { LVK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256 };
  uintptr_t guid = 0;
  HWDeviceType type = HWDeviceType_Software;
  char name[LVK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {0};
};

enum StorageType {
  StorageType_Device,
  StorageType_HostVisible,
  StorageType_Memoryless
};

enum CullMode : uint8_t { CullMode_None, CullMode_Front, CullMode_Back };
enum WindingMode : uint8_t { WindingMode_CCW, WindingMode_CW };

struct Result {
  enum class Code {
    Ok,
    ArgumentOutOfRange,
    RuntimeError,
  };

  Code code = Code::Ok;
  const char* message = "";
  explicit Result() = default;
  explicit Result(Code code, const char* message = "") : code(code), message(message) {}

  bool isOk() const {
    return code == Result::Code::Ok;
  }

  static void setResult(Result* outResult, Code code, const char* message = "") {
    if (outResult) {
      outResult->code = code;
      outResult->message = message;
    }
  }

  static void setResult(Result* outResult, const Result& sourceResult) {
    if (outResult) {
      *outResult = sourceResult;
    }
  }
};

struct ScissorRect {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct Dimensions {
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
  inline Dimensions divide1D(uint32_t v) const {
    return {.width = width / v, .height = height, .depth = depth};
  }
  inline Dimensions divide2D(uint32_t v) const {
    return {.width = width / v, .height = height / v, .depth = depth};
  }
  inline Dimensions divide3D(uint32_t v) const {
    return {.width = width / v, .height = height / v, .depth = depth / v};
  }
  inline bool operator==(const Dimensions& other) const {
    return width == other.width && height == other.height && depth == other.depth;
  }
};

struct Viewport {
  float x = 0.0f;
  float y = 0.0f;
  float width = 1.0f;
  float height = 1.0f;
  float minDepth = 0.0f;
  float maxDepth = 1.0f;
};

enum CompareOp : uint8_t {
  CompareOp_Never = 0,
  CompareOp_Less,
  CompareOp_Equal,
  CompareOp_LessEqual,
  CompareOp_Greater,
  CompareOp_NotEqual,
  CompareOp_GreaterEqual,
  CompareOp_AlwaysPass
};

enum StencilOp : uint8_t {
  StencilOp_Keep = 0,
  StencilOp_Zero,
  StencilOp_Replace,
  StencilOp_IncrementClamp,
  StencilOp_DecrementClamp,
  StencilOp_Invert,
  StencilOp_IncrementWrap,
  StencilOp_DecrementWrap
};

enum BlendOp : uint8_t {
  BlendOp_Add = 0,
  BlendOp_Subtract,
  BlendOp_ReverseSubtract,
  BlendOp_Min,
  BlendOp_Max
};

enum BlendFactor : uint8_t {
  BlendFactor_Zero = 0,
  BlendFactor_One,
  BlendFactor_SrcColor,
  BlendFactor_OneMinusSrcColor,
  BlendFactor_SrcAlpha,
  BlendFactor_OneMinusSrcAlpha,
  BlendFactor_DstColor,
  BlendFactor_OneMinusDstColor,
  BlendFactor_DstAlpha,
  BlendFactor_OneMinusDstAlpha,
  BlendFactor_SrcAlphaSaturated,
  BlendFactor_BlendColor,
  BlendFactor_OneMinusBlendColor,
  BlendFactor_BlendAlpha,
  BlendFactor_OneMinusBlendAlpha,
  BlendFactor_Src1Color,
  BlendFactor_OneMinusSrc1Color,
  BlendFactor_Src1Alpha,
  BlendFactor_OneMinusSrc1Alpha
};

struct SamplerStateDesc {
  SamplerFilter minFilter = SamplerFilter_Linear;
  SamplerFilter magFilter = SamplerFilter_Linear;
  SamplerMip mipMap = SamplerMip_Disabled;
  SamplerWrap wrapU = SamplerWrap_Repeat;
  SamplerWrap wrapV = SamplerWrap_Repeat;
  SamplerWrap wrapW = SamplerWrap_Repeat;
  CompareOp depthCompareOp = CompareOp_LessEqual;
  uint8_t mipLodMin = 0;
  uint8_t mipLodMax = 15;
  uint8_t maxAnisotropic = 1;
  bool depthCompareEnabled = false;
  const char* debugName = "";
};

struct StencilState {
  StencilOp stencilFailureOp = StencilOp_Keep;
  StencilOp depthFailureOp = StencilOp_Keep;
  StencilOp depthStencilPassOp = StencilOp_Keep;
  CompareOp stencilCompareOp = CompareOp_AlwaysPass;
  uint32_t readMask = (uint32_t)~0;
  uint32_t writeMask = (uint32_t)~0;
};

struct DepthState {
  CompareOp compareOp = CompareOp_AlwaysPass;
  bool isDepthWriteEnabled = false;
};

enum PolygonMode : uint8_t {
  PolygonMode_Fill = 0,
  PolygonMode_Line = 1,
};

enum class VertexFormat {
  Invalid = 0,

  Float1,
  Float2,
  Float3,
  Float4,

  Byte1,
  Byte2,
  Byte3,
  Byte4,

  UByte1,
  UByte2,
  UByte3,
  UByte4,

  Short1,
  Short2,
  Short3,
  Short4,

  UShort1,
  UShort2,
  UShort3,
  UShort4,

  Byte2Norm,
  Byte4Norm,

  UByte2Norm,
  UByte4Norm,

  Short2Norm,
  Short4Norm,

  UShort2Norm,
  UShort4Norm,

  Int1,
  Int2,
  Int3,
  Int4,

  UInt1,
  UInt2,
  UInt3,
  UInt4,

  HalfFloat1,
  HalfFloat2,
  HalfFloat3,
  HalfFloat4,

  Int_2_10_10_10_REV,
};

enum Format : uint8_t {
  Format_Invalid = 0,

  Format_R_UN8,
  Format_R_UI16,
  Format_R_UI32,
  Format_R_UN16,
  Format_R_F16,
  Format_R_F32,

  Format_RG_UN8,
  Format_RG_UI16,
  Format_RG_UI32,
  Format_RG_UN16,
  Format_RG_F16,
  Format_RG_F32,

  Format_RGBA_UN8,
  Format_RGBA_UI32,
  Format_RGBA_F16,
  Format_RGBA_F32,
  Format_RGBA_SRGB8,

  Format_BGRA_UN8,
  Format_BGRA_SRGB8,

  Format_A2B10G10R10_UN,
  Format_A2R10G10B10_UN,

  Format_ETC2_RGB8,
  Format_ETC2_SRGB8,
  Format_BC7_RGBA,

  Format_Z_UN16,
  Format_Z_UN24,
  Format_Z_F32,
  Format_Z_UN24_S_UI8,
  Format_Z_F32_S_UI8,

  Format_YUV_NV12,
  Format_YUV_420p,
};

enum LoadOp : uint8_t {
  LoadOp_Invalid = 0,
  LoadOp_DontCare,
  LoadOp_Load,
  LoadOp_Clear,
  LoadOp_None,
};

enum StoreOp : uint8_t {
  StoreOp_DontCare = 0,
  StoreOp_Store,
  StoreOp_MsaaResolve,
  StoreOp_None,
};

enum ResolveMode : uint8_t {
  ResolveMode_None = 0,
  ResolveMode_SampleZero, // always supported
  ResolveMode_Average,
  ResolveMode_Min,
  ResolveMode_Max,
};

enum ShaderStage : uint8_t {
  Stage_Vert,
  Stage_Tesc,
  Stage_Tese,
  Stage_Geom,
  Stage_Frag,
  Stage_Comp,
  Stage_Task,
  Stage_Mesh,
  // ray tracing
  Stage_RayGen,
  Stage_AnyHit,
  Stage_ClosestHit,
  Stage_Miss,
  Stage_Intersection,
  Stage_Callable,
};

union ClearColorValue {
  float float32[4];
  int32_t int32[4];
  uint32_t uint32[4];
};

struct VertexInput final {
  enum { LVK_VERTEX_ATTRIBUTES_MAX = 16 };
  enum { LVK_VERTEX_BUFFER_MAX = 16 };
  struct VertexAttribute final {
    uint32_t location = 0; // a buffer which contains this attribute stream
    uint32_t binding = 0;
    VertexFormat format = VertexFormat::Invalid; // per-element format
    uintptr_t offset = 0; // an offset where the first element of this attribute stream starts
  } attributes[LVK_VERTEX_ATTRIBUTES_MAX];
  struct VertexInputBinding final {
    uint32_t stride = 0;
  } inputBindings[LVK_VERTEX_BUFFER_MAX];

  uint32_t getNumAttributes() const {
    uint32_t n = 0;
    while (n < LVK_VERTEX_ATTRIBUTES_MAX && attributes[n].format != VertexFormat::Invalid) {
      n++;
    }
    return n;
  }
  uint32_t getNumInputBindings() const {
    uint32_t n = 0;
    while (n < LVK_VERTEX_BUFFER_MAX && inputBindings[n].stride) {
      n++;
    }
    return n;
  }
  uint32_t getVertexSize() const;

  bool operator==(const VertexInput& other) const {
    return memcmp(this, &other, sizeof(VertexInput)) == 0;
  }
};

struct ColorAttachment {
  Format format = Format_Invalid;
  bool blendEnabled = false;
  BlendOp rgbBlendOp = BlendOp::BlendOp_Add;
  BlendOp alphaBlendOp = BlendOp::BlendOp_Add;
  BlendFactor srcRGBBlendFactor = BlendFactor_One;
  BlendFactor srcAlphaBlendFactor = BlendFactor_One;
  BlendFactor dstRGBBlendFactor = BlendFactor_Zero;
  BlendFactor dstAlphaBlendFactor = BlendFactor_Zero;
};

struct ShaderModuleDesc {
  ShaderStage stage = Stage_Frag;
  const char* data = nullptr;
  size_t dataSize = 0; // if `dataSize` is non-zero, interpret `data` as binary shader data
  const char* debugName = "";

  ShaderModuleDesc(const char* source, lvk::ShaderStage stage, const char* debugName) : stage(stage), data(source), debugName(debugName) {}
  ShaderModuleDesc(const void* data, size_t dataLength, lvk::ShaderStage stage, const char* debugName) :
    stage(stage), data(static_cast<const char*>(data)), dataSize(dataLength), debugName(debugName) {
    LVK_ASSERT(dataSize);
  }
};

struct SpecializationConstantEntry {
  uint32_t constantId = 0;
  uint32_t offset = 0; // offset within SpecializationConstantDesc::data
  size_t size = 0;
};

struct SpecializationConstantDesc {
  enum { LVK_SPECIALIZATION_CONSTANTS_MAX = 16 };
  SpecializationConstantEntry entries[LVK_SPECIALIZATION_CONSTANTS_MAX] = {};
  const void* data = nullptr;
  size_t dataSize = 0;
  uint32_t getNumSpecializationConstants() const {
    uint32_t n = 0;
    while (n < LVK_SPECIALIZATION_CONSTANTS_MAX && entries[n].size) {
      n++;
    }
    return n;
  }
};

struct RenderPipelineDesc final {
  Topology topology = Topology_Triangle;

  lvk::VertexInput vertexInput;

  ShaderModuleHandle smVert;
  ShaderModuleHandle smTesc;
  ShaderModuleHandle smTese;
  ShaderModuleHandle smGeom;
  ShaderModuleHandle smTask;
  ShaderModuleHandle smMesh;
  ShaderModuleHandle smFrag;

  SpecializationConstantDesc specInfo = {};

  const char* entryPointVert = "main";
  const char* entryPointTesc = "main";
  const char* entryPointTese = "main";
  const char* entryPointGeom = "main";
  const char* entryPointTask = "main";
  const char* entryPointMesh = "main";
  const char* entryPointFrag = "main";

  ColorAttachment color[LVK_MAX_COLOR_ATTACHMENTS] = {};
  Format depthFormat = Format_Invalid;
  Format stencilFormat = Format_Invalid;

  CullMode cullMode = lvk::CullMode_None;
  WindingMode frontFaceWinding = lvk::WindingMode_CCW;
  PolygonMode polygonMode = lvk::PolygonMode_Fill;

  StencilState backFaceStencil = {};
  StencilState frontFaceStencil = {};

  uint32_t samplesCount = 1u;
  uint32_t patchControlPoints = 0;
  float minSampleShading = 0.0f;

  const char* debugName = "";

  uint32_t getNumColorAttachments() const {
    uint32_t n = 0;
    while (n < LVK_MAX_COLOR_ATTACHMENTS && color[n].format != Format_Invalid) {
      n++;
    }
    return n;
  }
};

struct ComputePipelineDesc final {
  ShaderModuleHandle smComp;
  SpecializationConstantDesc specInfo = {};
  const char* entryPoint = "main";
  const char* debugName = "";
};

struct RayTracingPipelineDesc final {
  ShaderModuleHandle smRayGen[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  ShaderModuleHandle smAnyHit[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  ShaderModuleHandle smClosestHit[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  ShaderModuleHandle smMiss[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  ShaderModuleHandle smIntersection[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  ShaderModuleHandle smCallable[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  SpecializationConstantDesc specInfo = {};
  const char* entryPoint = "main";
  const char* debugName = "";

#define GET_SHADER_GROUP_SIZE(name, module) \
  [[nodiscard]] uint32_t getShaderGroupSize##name() const { \
    uint32_t n = 0; \
    while (n < LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE && module[n]) n++; \
    return n; \
  }

  GET_SHADER_GROUP_SIZE(RayGen, smRayGen)
  GET_SHADER_GROUP_SIZE(AnyHit, smAnyHit)
  GET_SHADER_GROUP_SIZE(ClosestHit, smClosestHit)
  GET_SHADER_GROUP_SIZE(RayMiss, smMiss)
  GET_SHADER_GROUP_SIZE(Intersection, smIntersection)
  GET_SHADER_GROUP_SIZE(Callable, smCallable)

#undef GET_SHADER_GROUP_SIZE
};

struct RenderPass final {
  struct AttachmentDesc final {
    LoadOp loadOp = LoadOp_Invalid;
    StoreOp storeOp = StoreOp_Store;
    ResolveMode resolveMode = ResolveMode_Average;
    uint8_t layer = 0;
    uint8_t level = 0;
    ClearColorValue clearColor = {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}};
    float clearDepth = 1.0f;
    uint32_t clearStencil = 0;
  };

  AttachmentDesc color[LVK_MAX_COLOR_ATTACHMENTS] = {};
  AttachmentDesc depth = {.loadOp = LoadOp_DontCare, .storeOp = StoreOp_DontCare};
  AttachmentDesc stencil = {.loadOp = LoadOp_Invalid, .storeOp = StoreOp_DontCare};

  uint32_t layerCount = 1;
  uint32_t viewMask = 0;

  uint32_t getNumColorAttachments() const {
    uint32_t n = 0;
    while (n < LVK_MAX_COLOR_ATTACHMENTS && color[n].loadOp != LoadOp_Invalid) {
      n++;
    }
    return n;
  }
};

struct Framebuffer final {
  struct AttachmentDesc {
    TextureHandle texture;
    TextureHandle resolveTexture;
  };

  AttachmentDesc color[LVK_MAX_COLOR_ATTACHMENTS] = {};
  AttachmentDesc depthStencil;

  const char* debugName = "";

  uint32_t getNumColorAttachments() const {
    uint32_t n = 0;
    while (n < LVK_MAX_COLOR_ATTACHMENTS && color[n].texture) {
      n++;
    }
    return n;
  }
};

enum BufferUsageBits : uint8_t {
  BufferUsageBits_Index = 1 << 0,
  BufferUsageBits_Vertex = 1 << 1,
  BufferUsageBits_Uniform = 1 << 2,
  BufferUsageBits_Storage = 1 << 3,
  BufferUsageBits_Indirect = 1 << 4,
  // ray tracing
  BufferUsageBits_ShaderBindingTable = 1 << 5,
  BufferUsageBits_AccelStructBuildInputReadOnly = 1 << 6,
  BufferUsageBits_AccelStructStorage = 1 << 7
};

struct BufferDesc final {
  uint8_t usage = 0;
  StorageType storage = StorageType_HostVisible;
  size_t size = 0;
  const void* data = nullptr;
  const char* debugName = "";
};

struct Offset3D {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;
};

struct TextureLayers {
  uint32_t mipLevel = 0;
  uint32_t layer = 0;
  uint32_t numLayers = 1;
};

struct TextureRangeDesc {
  Offset3D offset = {};
  Dimensions dimensions = {1, 1, 1};
  uint32_t layer = 0;
  uint32_t numLayers = 1;
  uint32_t mipLevel = 0;
  uint32_t numMipLevels = 1;
};

enum TextureUsageBits : uint8_t {
  TextureUsageBits_Sampled = 1 << 0,
  TextureUsageBits_Storage = 1 << 1,
  TextureUsageBits_Attachment = 1 << 2,
};

enum Swizzle : uint8_t {
  Swizzle_Default = 0,
  Swizzle_0,
  Swizzle_1,
  Swizzle_R,
  Swizzle_G,
  Swizzle_B,
  Swizzle_A,
};

struct ComponentMapping {
  Swizzle r = Swizzle_Default;
  Swizzle g = Swizzle_Default;
  Swizzle b = Swizzle_Default;
  Swizzle a = Swizzle_Default;
  bool identity() const {
    return r == Swizzle_Default && g == Swizzle_Default && b == Swizzle_Default && a == Swizzle_Default;
  }
};

struct TextureDesc {
  TextureType type = TextureType_2D;
  Format format = Format_Invalid;

  Dimensions dimensions = {1, 1, 1};
  uint32_t numLayers = 1;
  uint32_t numSamples = 1;
  uint8_t usage = TextureUsageBits_Sampled;
  uint32_t numMipLevels = 1;
  StorageType storage = StorageType_Device;
  ComponentMapping swizzle = {};
  const void* data = nullptr;
  uint32_t dataNumMipLevels = 1; // how many mip-levels we want to upload
  bool generateMipmaps = false; // generate mip-levels immediately, valid only with non-null data
  const char* debugName = "";
};

struct TextureViewDesc {
  TextureType type = TextureType_2D;
  uint32_t layer = 0;
  uint32_t numLayers = 1;
  uint32_t mipLevel = 0;
  uint32_t numMipLevels = 1;
  ComponentMapping swizzle = {};
};

enum AccelStructType : uint8_t {
  AccelStructType_Invalid = 0,
  AccelStructType_TLAS = 1,
  AccelStructType_BLAS = 2,
};

enum AccelStructGeomType : uint8_t {
  AccelStructGeomType_Triangles = 0,
  AccelStructGeomType_AABBs = 1,
  AccelStructGeomType_Instances = 2,
};

enum AccelStructBuildFlagBits : uint8_t {
  AccelStructBuildFlagBits_AllowUpdate = 1 << 0,
  AccelStructBuildFlagBits_AllowCompaction = 1 << 1,
  AccelStructBuildFlagBits_PreferFastTrace = 1 << 2,
  AccelStructBuildFlagBits_PreferFastBuild = 1 << 3,
  AccelStructBuildFlagBits_LowMemory = 1 << 4,
};

enum AccelStructGeometryFlagBits : uint8_t {
  AccelStructGeometryFlagBits_Opaque = 1 << 0,
  AccelStructGeometryFlagBits_NoDuplicateAnyHit = 1 << 1,
};

enum AccelStructInstanceFlagBits : uint8_t {
  AccelStructInstanceFlagBits_TriangleFacingCullDisable = 1 << 0,
  AccelStructInstanceFlagBits_TriangleFlipFacing = 1 << 1,
  AccelStructInstanceFlagBits_ForceOpaque = 1 << 2,
  AccelStructInstanceFlagBits_ForceNoOpaque = 1 << 3,
};

struct AccelStructSizes {
  uint64_t accelerationStructureSize = 0;
  uint64_t updateScratchSize = 0;
  uint64_t buildScratchSize = 0;
};

struct AccelStructBuildRange {
  uint32_t primitiveCount = 0;
  uint32_t primitiveOffset = 0;
  uint32_t firstVertex = 0;
  uint32_t transformOffset = 0;
};

struct mat3x4 {
  float matrix[3][4];
};

struct AccelStructInstance {
  mat3x4 transform;
  uint32_t instanceCustomIndex : 24 = 0;
  uint32_t mask : 8 = 0xff;
  uint32_t instanceShaderBindingTableRecordOffset : 24 = 0;
  uint32_t flags : 8 = AccelStructInstanceFlagBits_TriangleFacingCullDisable;
  uint64_t accelerationStructureReference = 0;
};

struct AccelStructDesc {
  AccelStructType type = AccelStructType_Invalid;
  AccelStructGeomType geometryType = AccelStructGeomType_Triangles;
  uint8_t geometryFlags = AccelStructGeometryFlagBits_Opaque;

  VertexFormat vertexFormat = VertexFormat::Invalid;
  BufferHandle vertexBuffer;
  uint32_t vertexStride = 0; // zero means the size of `vertexFormat`
  uint32_t numVertices = 0;
  IndexFormat indexFormat = IndexFormat_UI32;
  BufferHandle indexBuffer;
  BufferHandle transformBuffer;
  BufferHandle instancesBuffer;
  AccelStructBuildRange buildRange = {};
  uint8_t buildFlags = AccelStructBuildFlagBits_PreferFastTrace;
  const char* debugName = "";
};

struct Dependencies {
  enum { LVK_MAX_SUBMIT_DEPENDENCIES = 4 };
  TextureHandle textures[LVK_MAX_SUBMIT_DEPENDENCIES] = {};
  BufferHandle buffers[LVK_MAX_SUBMIT_DEPENDENCIES] = {};
};

class ICommandBuffer {
 public:
  virtual ~ICommandBuffer() = default;

  virtual void transitionToShaderReadOnly(TextureHandle surface) const = 0;

  virtual void cmdPushDebugGroupLabel(const char* label, uint32_t colorRGBA = 0xffffffff) const = 0;
  virtual void cmdInsertDebugEventLabel(const char* label, uint32_t colorRGBA = 0xffffffff) const = 0;
  virtual void cmdPopDebugGroupLabel() const = 0;

  virtual void cmdBindRayTracingPipeline(lvk::RayTracingPipelineHandle handle) = 0;

  virtual void cmdBindComputePipeline(lvk::ComputePipelineHandle handle) = 0;
  virtual void cmdDispatchThreadGroups(const Dimensions& threadgroupCount, const Dependencies& deps = {}) = 0;
  virtual void cmdDispatchThreadGroupsIndirect(BufferHandle indirectBuffer, size_t indirectBufferOffset, const Dependencies& deps = {}) = 0;

  virtual void cmdBeginRendering(const lvk::RenderPass& renderPass, const lvk::Framebuffer& desc, const Dependencies& deps = {}) = 0;
  virtual void cmdEndRendering() = 0;

  virtual void cmdBindViewport(const Viewport& viewport) = 0;
  virtual void cmdBindScissorRect(const ScissorRect& rect) = 0;

  virtual void cmdBindRenderPipeline(lvk::RenderPipelineHandle handle) = 0;
  virtual void cmdBindDepthState(const DepthState& state) = 0;

  virtual void cmdBindVertexBuffer(uint32_t index, BufferHandle buffer, uint64_t bufferOffset = 0) = 0;
  virtual void cmdBindIndexBuffer(BufferHandle indexBuffer, IndexFormat indexFormat, uint64_t indexBufferOffset = 0) = 0;
  virtual void cmdPushConstants(const void* data, size_t size, size_t offset = 0) = 0;
  template<typename Struct>
  void cmdPushConstants(const Struct& data, size_t offset = 0) {
    this->cmdPushConstants(&data, sizeof(Struct), offset);
  }

  virtual void cmdFillBuffer(BufferHandle buffer, size_t bufferOffset, size_t size, uint32_t data) = 0;
  virtual void cmdUpdateBuffer(BufferHandle buffer, size_t bufferOffset, size_t size, const void* data) = 0;
  template<typename Struct>
  void cmdUpdateBuffer(BufferHandle buffer, const Struct& data, size_t bufferOffset = 0) {
    this->cmdUpdateBuffer(buffer, bufferOffset, sizeof(Struct), &data);
  }

  virtual void cmdDraw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0, uint32_t baseInstance = 0) = 0;
  virtual void cmdDrawIndexed(uint32_t indexCount,
                              uint32_t instanceCount = 1,
                              uint32_t firstIndex = 0,
                              int32_t vertexOffset = 0,
                              uint32_t baseInstance = 0) = 0;
  virtual void cmdDrawIndirect(BufferHandle indirectBuffer, size_t indirectBufferOffset, uint32_t drawCount, uint32_t stride = 0) = 0;
  virtual void cmdDrawIndexedIndirect(BufferHandle indirectBuffer,
                                      size_t indirectBufferOffset,
                                      uint32_t drawCount,
                                      uint32_t stride = 0) = 0;
  virtual void cmdDrawIndexedIndirectCount(BufferHandle indirectBuffer,
                                           size_t indirectBufferOffset,
                                           BufferHandle countBuffer,
                                           size_t countBufferOffset,
                                           uint32_t maxDrawCount,
                                           uint32_t stride = 0) = 0;
  virtual void cmdDrawMeshTasks(const Dimensions& threadgroupCount) = 0;
  virtual void cmdDrawMeshTasksIndirect(BufferHandle indirectBuffer,
                                        size_t indirectBufferOffset,
                                        uint32_t drawCount,
                                        uint32_t stride = 0) = 0;
  virtual void cmdDrawMeshTasksIndirectCount(BufferHandle indirectBuffer,
                                             size_t indirectBufferOffset,
                                             BufferHandle countBuffer,
                                             size_t countBufferOffset,
                                             uint32_t maxDrawCount,
                                             uint32_t stride = 0) = 0;
  virtual void cmdTraceRays(uint32_t width, uint32_t height, uint32_t depth = 1, const Dependencies& deps = {}) = 0;

  virtual void cmdSetBlendColor(const float color[4]) = 0;
  virtual void cmdSetDepthBias(float constantFactor, float slopeFactor, float clamp = 0.0f) = 0;
  virtual void cmdSetDepthBiasEnable(bool enable) = 0;

  virtual void cmdResetQueryPool(QueryPoolHandle pool, uint32_t firstQuery, uint32_t queryCount) = 0;
  virtual void cmdWriteTimestamp(QueryPoolHandle pool, uint32_t query) = 0;

  virtual void cmdClearColorImage(TextureHandle tex, const ClearColorValue& value, const TextureLayers& layers = {}) = 0;
  virtual void cmdCopyImage(TextureHandle src,
                            TextureHandle dst,
                            const Dimensions& extent,
                            const Offset3D& srcOffset = {},
                            const Offset3D& dstOffset = {},
                            const TextureLayers& srcLayers = {},
                            const TextureLayers& dstLayers = {}) = 0;
  virtual void cmdGenerateMipmap(TextureHandle handle) = 0;
  virtual void cmdUpdateTLAS(AccelStructHandle handle, BufferHandle instancesBuffer) = 0;
};

struct SubmitHandle {
  uint32_t bufferIndex_ = 0;
  uint32_t submitId_ = 0;
  SubmitHandle() = default;
  explicit SubmitHandle(uint64_t handle) : bufferIndex_(uint32_t(handle & 0xffffffff)), submitId_(uint32_t(handle >> 32)) {
    LVK_ASSERT(submitId_);
  }
  bool empty() const {
    return submitId_ == 0;
  }
  uint64_t handle() const {
    return (uint64_t(submitId_) << 32) + bufferIndex_;
  }
};

static_assert(sizeof(SubmitHandle) == sizeof(uint64_t));

class IContext {
 protected:
  IContext() = default;

 public:
  virtual ~IContext() = default;

  virtual ICommandBuffer& acquireCommandBuffer() = 0;

  virtual SubmitHandle submit(ICommandBuffer& commandBuffer, TextureHandle present = {}) = 0;
  virtual void wait(SubmitHandle handle) = 0; // waiting on an empty handle results in vkDeviceWaitIdle()

  [[nodiscard]] virtual Holder<BufferHandle> createBuffer(const BufferDesc& desc,
                                                          const char* debugName = nullptr,
                                                          Result* outResult = nullptr) = 0;
  [[nodiscard]] virtual Holder<SamplerHandle> createSampler(const SamplerStateDesc& desc, Result* outResult = nullptr) = 0;
  [[nodiscard]] virtual Holder<TextureHandle> createTexture(const TextureDesc& desc,
                                                            const char* debugName = nullptr,
                                                            Result* outResult = nullptr) = 0;
  [[nodiscard]] virtual Holder<TextureHandle> createTextureView(TextureHandle texture,
                                                                const TextureViewDesc& desc,
                                                                const char* debugName = nullptr,
                                                                Result* outResult = nullptr) = 0;
  [[nodiscard]] virtual Holder<ComputePipelineHandle> createComputePipeline(const ComputePipelineDesc& desc,
                                                                            Result* outResult = nullptr) = 0;
  [[nodiscard]] virtual Holder<RenderPipelineHandle> createRenderPipeline(const RenderPipelineDesc& desc, Result* outResult = nullptr) = 0;
  [[nodiscard]] virtual Holder<RayTracingPipelineHandle> createRayTracingPipeline(const RayTracingPipelineDesc& desc,
                                                                            Result* outResult = nullptr) = 0;
  [[nodiscard]] virtual Holder<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc& desc, Result* outResult = nullptr) = 0;

  [[nodiscard]] virtual Holder<QueryPoolHandle> createQueryPool(uint32_t numQueries,
                                                                const char* debugName,
                                                                Result* outResult = nullptr) = 0;

  [[nodiscard]] virtual Holder<AccelStructHandle> createAccelerationStructure(const AccelStructDesc& desc, Result* outResult = nullptr) = 0;

  virtual void destroy(ComputePipelineHandle handle) = 0;
  virtual void destroy(RenderPipelineHandle handle) = 0;
  virtual void destroy(RayTracingPipelineHandle) = 0;
  virtual void destroy(ShaderModuleHandle handle) = 0;
  virtual void destroy(SamplerHandle handle) = 0;
  virtual void destroy(BufferHandle handle) = 0;
  virtual void destroy(TextureHandle handle) = 0;
  virtual void destroy(QueryPoolHandle handle) = 0;
  virtual void destroy(AccelStructHandle handle) = 0;
  virtual void destroy(Framebuffer& fb) = 0;

  [[nodiscard]] virtual uint64_t gpuAddress(AccelStructHandle handle) const = 0;

#pragma region Acceleration structure functions
  [[nodiscard]] virtual AccelStructSizes getAccelStructSizes(const AccelStructDesc& desc, Result* outResult = nullptr) const = 0;
#pragma endregion

#pragma region Buffer functions
  virtual Result upload(BufferHandle handle, const void* data, size_t size, size_t offset = 0) = 0;
  virtual Result download(BufferHandle handle, void* data, size_t size, size_t offset) = 0;
  [[nodiscard]] virtual uint8_t* getMappedPtr(BufferHandle handle) const = 0;
  [[nodiscard]] virtual uint64_t gpuAddress(BufferHandle handle, size_t offset = 0) const = 0;
  virtual void flushMappedMemory(BufferHandle handle, size_t offset, size_t size) const = 0;
  [[nodiscard]] virtual uint32_t getMaxStorageBufferRange() const = 0;
#pragma endregion

#pragma region Texture functions
  // `data` contains mip-levels and layers as in https://registry.khronos.org/KTX/specs/1.0/ktxspec.v1.html
  virtual Result upload(TextureHandle handle, const TextureRangeDesc& range, const void* data, uint32_t bufferRowLength = 0) = 0;
  virtual Result download(TextureHandle handle, const TextureRangeDesc& range, void* outData) = 0;
  [[nodiscard]] virtual Dimensions getDimensions(TextureHandle handle) const = 0;
  [[nodiscard]] virtual float getAspectRatio(TextureHandle handle) const = 0;
  [[nodiscard]] virtual Format getFormat(TextureHandle handle) const = 0;
#pragma endregion

  virtual TextureHandle getCurrentSwapchainTexture() = 0;
  virtual Format getSwapchainFormat() const = 0;
  virtual ColorSpace getSwapchainColorSpace() const = 0;
  virtual uint32_t getSwapchainCurrentImageIndex() const = 0;
  virtual uint32_t getNumSwapchainImages() const = 0;
  virtual void recreateSwapchain(int newWidth, int newHeight) = 0;
  
  // MSAA level is supported if ((samples & bitmask) != 0), where samples must be power of two.
  virtual uint32_t getFramebufferMSAABitMask() const = 0;

#pragma region Performance queries
  virtual double getTimestampPeriodToMs() const = 0;
  virtual bool getQueryPoolResults(QueryPoolHandle pool,
                                   uint32_t firstQuery,
                                   uint32_t queryCount,
                                   size_t dataSize,
                                   void* outData,
                                   size_t stride) const = 0;
#pragma endregion
};

} // namespace lvk

#if LVK_WITH_GLFW
typedef struct GLFWwindow GLFWwindow;
#endif

namespace lvk {

using ShaderModuleErrorCallback = void (*)(lvk::IContext*, lvk::ShaderModuleHandle, int line, int col, const char* debugName);

constexpr uint32_t kMaxCustomExtensions = 32;

enum VulkanVersion {
  VulkanVersion_1_3,
  VulkanVersion_1_4,
};

struct ContextConfig {
  VulkanVersion vulkanVersion = VulkanVersion_1_3;
  bool terminateOnValidationError = false; // invoke std::terminate() on any validation error
  bool enableValidation = true;
  lvk::ColorSpace swapchainRequestedColorSpace = lvk::ColorSpace_SRGB_NONLINEAR;
  // owned by the application - should be alive until createVulkanContextWithSwapchain() returns
  const void* pipelineCacheData = nullptr;
  size_t pipelineCacheDataSize = 0;
  ShaderModuleErrorCallback shaderModuleErrorCallback = nullptr;
  const char* extensionsInstance[kMaxCustomExtensions] = {}; // add extra instance extensions on top of required ones
  const char* extensionsDevice[kMaxCustomExtensions] = {}; // add extra device extensions on top of required ones
  void* extensionsDeviceFeatures = nullptr; // inserted into VkPhysicalDeviceVulkan11Features::pNext

  // LVK knows about these extensions and can manage them automatically upon request
  bool enableHeadlessSurface = false; // VK_EXT_headless_surface

  uint64_t maxStagingBufferSize = 128ull * 1024ull * 1024ull; // a reasonable default
};

[[nodiscard]] bool isDepthOrStencilFormat(lvk::Format format);
[[nodiscard]] uint32_t getNumImagePlanes(lvk::Format format);
[[nodiscard]] uint32_t getTextureBytesPerLayer(uint32_t width, uint32_t height, lvk::Format format, uint32_t level);
[[nodiscard]] uint32_t getTextureBytesPerPlane(uint32_t width, uint32_t height, lvk::Format format, uint32_t plane);
[[nodiscard]] uint32_t getVertexFormatSize(lvk::VertexFormat format);
void logShaderSource(const char* text);

constexpr uint32_t calcNumMipLevels(uint32_t width, uint32_t height) {
  uint32_t levels = 1;

  while ((width | height) >> levels)
    levels++;

  return levels;
}

#if LVK_WITH_GLFW
/*
 * width/height  > 0: window size in pixels
 * width/height == 0: take the whole monitor work area
 * width/height  < 0: take a percentage of the monitor work area, for example (-95, -90)
 *   The actual values in pixels are returned in parameters.
 */
GLFWwindow* initWindow(const char* windowTitle, int& outWidth, int& outHeight, bool resizable = false);
#endif // LVK_WITH_GLFW

#if LVK_WITH_GLFW || defined(ANDROID)
#if defined(ANDROID)
using LVKwindow = ANativeWindow;
#else
using LVKwindow = GLFWwindow;
#endif
std::unique_ptr<lvk::IContext> createVulkanContextWithSwapchain(LVKwindow* window,
                                                                uint32_t width,
                                                                uint32_t height,
                                                                const lvk::ContextConfig& cfg,
                                                                lvk::HWDeviceType preferredDeviceType = lvk::HWDeviceType_Discrete);
#endif // LVK_WITH_GLFW || defined(ANDROID)

} // namespace lvk
