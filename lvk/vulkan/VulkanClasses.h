﻿/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <lvk/vulkan/VulkanUtils.h>

#include <unordered_map>

namespace lvk {

class VulkanContext;

struct DeviceQueues {
  const static uint32_t INVALID = 0xFFFFFFFF;
  uint32_t graphicsQueueFamilyIndex = INVALID;
  uint32_t computeQueueFamilyIndex = INVALID;

  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue computeQueue = VK_NULL_HANDLE;
};

class VulkanBuffer final {
 public:
  VulkanBuffer() = default;
  VulkanBuffer(lvk::VulkanContext* ctx,
               VkDevice device,
               VkDeviceSize bufferSize,
               VkBufferUsageFlags usageFlags,
               VkMemoryPropertyFlags memFlags,
               const char* debugName = nullptr);
  ~VulkanBuffer();

  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;

  VulkanBuffer(VulkanBuffer&& other);
  VulkanBuffer& operator=(VulkanBuffer&& other);

  void bufferSubData(size_t offset, size_t size, const void* data);
  void getBufferSubData(size_t offset, size_t size, void* data);
  [[nodiscard]] uint8_t* getMappedPtr() const {
    return static_cast<uint8_t*>(mappedPtr_);
  }
  bool isMapped() const {
    return mappedPtr_ != nullptr;
  }
  void flushMappedMemory(VkDeviceSize offset, VkDeviceSize size) const;

 public:
  lvk::VulkanContext* ctx_ = nullptr;
  VkDevice device_ = VK_NULL_HANDLE;
  VkBuffer vkBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory vkMemory_ = VK_NULL_HANDLE;
  VmaAllocationCreateInfo vmaAllocInfo_ = {};
  VmaAllocation vmaAllocation_ = VK_NULL_HANDLE;
  VkDeviceAddress vkDeviceAddress_ = 0;
  VkDeviceSize bufferSize_ = 0;
  VkBufferUsageFlags vkUsageFlags_ = 0;
  VkMemoryPropertyFlags vkMemFlags_ = 0;
  void* mappedPtr_ = nullptr;
};

class VulkanImage final {
 public:
  VulkanImage(lvk::VulkanContext& ctx,
              VkDevice device,
              VkExtent3D extent,
              VkImageType type,
              VkFormat format,
              uint32_t numLevels,
              uint32_t numLayers,
              VkImageTiling tiling,
              VkImageUsageFlags usageFlags,
              VkMemoryPropertyFlags memFlags,
              VkImageCreateFlags createFlags,
              VkSampleCountFlagBits samples,
              const char* debugName);
  VulkanImage(lvk::VulkanContext& ctx,
              VkDevice device,
              VkImage image,
              VkImageUsageFlags usageFlags,
              VkFormat imageFormat,
              VkExtent3D extent,
              const char* debugName);
  ~VulkanImage();

  VulkanImage(const VulkanImage&) = delete;
  VulkanImage& operator=(const VulkanImage&) = delete;

  // clang-format off
  bool isSampledImage() const { return (vkUsageFlags_ & VK_IMAGE_USAGE_SAMPLED_BIT) > 0; }
  bool isStorageImage() const { return (vkUsageFlags_ & VK_IMAGE_USAGE_STORAGE_BIT) > 0; }
  // clang-format on

  /*
   * Setting `numLevels` to a non-zero value will override `mipLevels_` value from the original Vulkan image, and can be used to create
   * image views with different number of levels.
   */
  VkImageView createImageView(VkImageViewType type,
                              VkFormat format,
                              VkImageAspectFlags aspectMask,
                              uint32_t baseLevel,
                              uint32_t numLevels = VK_REMAINING_MIP_LEVELS,
                              uint32_t baseLayer = 0,
                              uint32_t numLayers = 1,
                              const char* debugName = nullptr) const;

  void generateMipmap(VkCommandBuffer commandBuffer) const;

  void transitionLayout(VkCommandBuffer commandBuffer,
                        VkImageLayout newImageLayout,
                        VkPipelineStageFlags srcStageMask,
                        VkPipelineStageFlags dstStageMask,
                        const VkImageSubresourceRange& subresourceRange) const;

  VkImageAspectFlags getImageAspectFlags() const;

  static bool isDepthFormat(VkFormat format);
  static bool isStencilFormat(VkFormat format);

 public:
  lvk::VulkanContext& ctx_;
  VkDevice vkDevice_ = VK_NULL_HANDLE;
  VkImage vkImage_ = VK_NULL_HANDLE;
  VkImageUsageFlags vkUsageFlags_ = 0;
  VkDeviceMemory vkMemory_ = VK_NULL_HANDLE;
  VmaAllocationCreateInfo vmaAllocInfo_ = {};
  VmaAllocation vmaAllocation_ = VK_NULL_HANDLE;
  VkFormatProperties vkFormatProperties_ = {};
  VkExtent3D vkExtent_ = {0, 0, 0};
  VkImageType vkType_ = VK_IMAGE_TYPE_MAX_ENUM;
  VkFormat vkImageFormat_ = VK_FORMAT_UNDEFINED;
  VkSampleCountFlagBits vkSamples_ = VK_SAMPLE_COUNT_1_BIT;
  void* mappedPtr_ = nullptr;
  bool isSwapchainImage_ = false;
  uint32_t numLevels_ = 1u;
  uint32_t numLayers_ = 1u;
  bool isDepthFormat_ = false;
  bool isStencilFormat_ = false;
  // current image layout
  mutable VkImageLayout vkImageLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct VulkanTexture final {
  VulkanTexture() = default;
  VulkanTexture(std::shared_ptr<lvk::VulkanImage> image, VkImageView imageView);
  ~VulkanTexture();

  VulkanTexture(const VulkanTexture&) = delete;
  VulkanTexture& operator=(const VulkanTexture&) = delete;

  VulkanTexture(VulkanTexture&& other);
  VulkanTexture& operator=(VulkanTexture&& other);

  VkExtent3D getExtent() const {
    LVK_ASSERT(image_.get());
    return image_->vkExtent_;
  }

  // framebuffers can render only into one level/layer
  VkImageView getOrCreateVkImageViewForFramebuffer(uint8_t level);

  std::shared_ptr<lvk::VulkanImage> image_;
  VkImageView imageView_ = VK_NULL_HANDLE; // all mip-levels
  VkImageView imageViewForFramebuffer_[LVK_MAX_MIP_LEVELS] = {};
};

class VulkanSwapchain final {
  enum { LVK_MAX_SWAPCHAIN_IMAGES = 16 };

 public:
  VulkanSwapchain(VulkanContext& ctx, uint32_t width, uint32_t height);
  ~VulkanSwapchain();

  Result present(VkSemaphore waitSemaphore);
  VkImage getCurrentVkImage() const;
  VkImageView getCurrentVkImageView() const;
  TextureHandle getCurrentTexture();

 public:
  VkSemaphore acquireSemaphore_ = VK_NULL_HANDLE;

 private:
  VulkanContext& ctx_;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue graphicsQueue_ = VK_NULL_HANDLE;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t numSwapchainImages_ = 0;
  uint32_t currentImageIndex_ = 0;
  bool getNextImage_ = true;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkSurfaceFormatKHR surfaceFormat_ = {.format = VK_FORMAT_UNDEFINED};
  TextureHandle swapchainTextures_[LVK_MAX_SWAPCHAIN_IMAGES] = {};
};

class VulkanImmediateCommands final {
 public:
  // the maximum number of command buffers which can similtaneously exist in the system; when we run out of buffers, we stall and wait until
  // an existing buffer becomes available
  static constexpr uint32_t kMaxCommandBuffers = 64;

  VulkanImmediateCommands(VkDevice device, uint32_t queueFamilyIndex, const char* debugName);
  ~VulkanImmediateCommands();
  VulkanImmediateCommands(const VulkanImmediateCommands&) = delete;
  VulkanImmediateCommands& operator=(const VulkanImmediateCommands&) = delete;

  struct CommandBufferWrapper {
    VkCommandBuffer cmdBuf_ = VK_NULL_HANDLE;
    VkCommandBuffer cmdBufAllocated_ = VK_NULL_HANDLE;
    SubmitHandle handle_ = {};
    VkFence fence_ = VK_NULL_HANDLE;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
    bool isEncoding_ = false;
  };

  // returns the current command buffer (creates one if it does not exist)
  const CommandBufferWrapper& acquire();
  SubmitHandle submit(const CommandBufferWrapper& wrapper);
  void waitSemaphore(VkSemaphore semaphore);
  VkSemaphore acquireLastSubmitSemaphore();
  SubmitHandle getLastSubmitHandle() const;
  bool isReady(SubmitHandle handle, bool fastCheckNoVulkan = false) const;
  void wait(SubmitHandle handle);
  void waitAll();

 private:
  void purge();

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex_ = 0;
  const char* debugName_ = "";
  CommandBufferWrapper buffers_[kMaxCommandBuffers];
  SubmitHandle lastSubmitHandle_ = SubmitHandle();
  VkSemaphore lastSubmitSemaphore_ = VK_NULL_HANDLE;
  VkSemaphore waitSemaphore_ = VK_NULL_HANDLE;
  uint32_t numAvailableCommandBuffers_ = kMaxCommandBuffers;
  uint32_t submitCounter_ = 1;
};

struct RenderPipelineDynamicState final {
  VkPrimitiveTopology topology_ : 4 = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  VkBool32 depthBiasEnable_ : 1 = VK_FALSE;

  void setTopology(VkPrimitiveTopology topology) {
    LVK_ASSERT_MSG(topology <= VK_PRIMITIVE_TOPOLOGY_PATCH_LIST, "Invalid VkPrimitiveTopology");
    topology_ = topology;
  }
};

static_assert(sizeof(RenderPipelineDynamicState) == sizeof(uint32_t));

struct RenderPipelineState final {
  void destroyPipelines(lvk::VulkanContext * ctx);

  RenderPipelineDesc desc_;

  uint32_t numBindings_ = 0;
  uint32_t numAttributes_ = 0;
  VkVertexInputBindingDescription vkBindings_[VertexInput::LVK_VERTEX_BUFFER_MAX] = {};
  VkVertexInputAttributeDescription vkAttributes_[VertexInput::LVK_VERTEX_ATTRIBUTES_MAX] = {};

  // non-owning, cached the last pipeline layout from the context
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;

  // [topology][depthBiasEnable]
  VkPipeline pipelines_[VK_PRIMITIVE_TOPOLOGY_PATCH_LIST + 1][2] = {};
};

class VulkanPipelineBuilder final {
 public:
  VulkanPipelineBuilder();
  ~VulkanPipelineBuilder() = default;

  VulkanPipelineBuilder& depthBiasEnable(bool enable);
  VulkanPipelineBuilder& dynamicState(VkDynamicState state);
  VulkanPipelineBuilder& primitiveTopology(VkPrimitiveTopology topology);
  VulkanPipelineBuilder& rasterizationSamples(VkSampleCountFlagBits samples);
  VulkanPipelineBuilder& shaderStage(VkPipelineShaderStageCreateInfo stage);
  VulkanPipelineBuilder& stencilStateOps(VkStencilFaceFlags faceMask,
                                         VkStencilOp failOp,
                                         VkStencilOp passOp,
                                         VkStencilOp depthFailOp,
                                         VkCompareOp compareOp);
  VulkanPipelineBuilder& stencilMasks(VkStencilFaceFlags faceMask, uint32_t compareMask, uint32_t writeMask, uint32_t reference);
  VulkanPipelineBuilder& cullMode(VkCullModeFlags mode);
  VulkanPipelineBuilder& frontFace(VkFrontFace mode);
  VulkanPipelineBuilder& polygonMode(VkPolygonMode mode);
  VulkanPipelineBuilder& vertexInputState(const VkPipelineVertexInputStateCreateInfo& state);
  VulkanPipelineBuilder& colorAttachments(const VkPipelineColorBlendAttachmentState* states,
                                          const VkFormat* formats,
                                          uint32_t numColorAttachments);
  VulkanPipelineBuilder& depthAttachmentFormat(VkFormat format);
  VulkanPipelineBuilder& stencilAttachmentFormat(VkFormat format);

  VkResult build(VkDevice device,
                 VkPipelineCache pipelineCache,
                 VkPipelineLayout pipelineLayout,
                 VkPipeline* outPipeline,
                 const char* debugName = nullptr) noexcept;

  static uint32_t getNumPipelinesCreated() {
    return numPipelinesCreated_;
  }

 private:
  enum { LVK_MAX_DYNAMIC_STATES = 128 };
  uint32_t numDynamicStates_ = 0;
  VkDynamicState dynamicStates_[LVK_MAX_DYNAMIC_STATES] = {};

  uint32_t numShaderStages_ = 0;
  VkPipelineShaderStageCreateInfo shaderStages_[Stage_Frag + 1] = {};

  VkPipelineVertexInputStateCreateInfo vertexInputState_;
  VkPipelineInputAssemblyStateCreateInfo inputAssembly_;
  VkPipelineRasterizationStateCreateInfo rasterizationState_;
  VkPipelineMultisampleStateCreateInfo multisampleState_;
  VkPipelineDepthStencilStateCreateInfo depthStencilState_;

  uint32_t numColorAttachments_ = 0;
  VkPipelineColorBlendAttachmentState colorBlendAttachmentStates_[LVK_MAX_COLOR_ATTACHMENTS] = {};
  VkFormat colorAttachmentFormats_[LVK_MAX_COLOR_ATTACHMENTS] = {};

  VkFormat depthAttachmentFormat_ = VK_FORMAT_UNDEFINED;
  VkFormat stencilAttachmentFormat_ = VK_FORMAT_UNDEFINED;

  static uint32_t numPipelinesCreated_;
};

struct ComputePipelineState final {
  ComputePipelineDesc desc_;
  // non-owning, cached the last pipeline layout from the context
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class CommandBuffer final : public ICommandBuffer {
 public:
  CommandBuffer() = default;
  explicit CommandBuffer(VulkanContext* ctx);
  ~CommandBuffer() override;

  CommandBuffer& operator=(CommandBuffer&& other) = default;

  void transitionToShaderReadOnly(TextureHandle surface) const override;

  void cmdBindComputePipeline(lvk::ComputePipelineHandle handle) override;
  void cmdDispatchThreadGroups(const Dimensions& threadgroupCount, const Dependencies& deps) override;

  void cmdPushDebugGroupLabel(const char* label, uint32_t colorRGBA) const override;
  void cmdInsertDebugEventLabel(const char* label, uint32_t colorRGBA) const override;
  void cmdPopDebugGroupLabel() const override;

  void cmdBeginRendering(const lvk::RenderPass& renderPass, const lvk::Framebuffer& desc) override;
  void cmdEndRendering() override;

  void cmdBindViewport(const Viewport& viewport) override;
  void cmdBindScissorRect(const ScissorRect& rect) override;

  void cmdBindRenderPipeline(lvk::RenderPipelineHandle handle) override;
  void cmdBindDepthState(const DepthState& state) override;

  void cmdBindVertexBuffer(uint32_t index, BufferHandle buffer, size_t bufferOffset) override;
  void cmdBindIndexBuffer(BufferHandle indexBuffer, IndexFormat indexFormat, size_t indexBufferOffset) override;
  void cmdPushConstants(const void* data, size_t size, size_t offset) override;

  void cmdDraw(PrimitiveType primitiveType, size_t vertexStart, size_t vertexCount) override;
  void cmdDrawIndexed(PrimitiveType primitiveType,
                      uint32_t indexCount,
                      uint32_t instanceCount,
                      uint32_t firstIndex,
                      int32_t vertexOffset,
                      uint32_t baseInstance) override;
  void cmdDrawIndirect(PrimitiveType primitiveType,
                       BufferHandle indirectBuffer,
                       size_t indirectBufferOffset,
                       uint32_t drawCount,
                       uint32_t stride = 0) override;
  void cmdDrawIndexedIndirect(PrimitiveType primitiveType,
                              BufferHandle indirectBuffer,
                              size_t indirectBufferOffset,
                              uint32_t drawCount,
                              uint32_t stride = 0) override;

  void cmdSetBlendColor(const float color[4]) override;
  void cmdSetDepthBias(float depthBias, float slopeScale, float clamp) override;

 private:
  void useComputeTexture(TextureHandle texture);
  void bindGraphicsPipeline();

 private:
  friend class VulkanContext;

  VulkanContext* ctx_ = nullptr;
  const VulkanImmediateCommands::CommandBufferWrapper* wrapper_ = nullptr;

  lvk::Framebuffer framebuffer_ = {};
  lvk::SubmitHandle lastSubmitHandle_ = {};

  VkPipeline lastPipelineBound_ = VK_NULL_HANDLE;

  bool isRendering_ = false;

  lvk::RenderPipelineHandle currentPipeline_ = {};
  lvk::RenderPipelineDynamicState dynamicState_ = {};
};

class VulkanStagingDevice final {
 public:
  explicit VulkanStagingDevice(VulkanContext& ctx);
  ~VulkanStagingDevice();

  VulkanStagingDevice(const VulkanStagingDevice&) = delete;
  VulkanStagingDevice& operator=(const VulkanStagingDevice&) = delete;

  void bufferSubData(VulkanBuffer& buffer, size_t dstOffset, size_t size, const void* data);
  void imageData2D(VulkanImage& image,
                   const VkRect2D& imageRegion,
                   uint32_t baseMipLevel,
                   uint32_t numMipLevels,
                   uint32_t layer,
                   uint32_t numLayers,
                   VkFormat format,
                   const void* data[]);
  void imageData3D(VulkanImage& image, const VkOffset3D& offset, const VkExtent3D& extent, VkFormat format, const void* data);

 private:
  struct MemoryRegionDesc {
    uint32_t srcOffset_ = 0;
    uint32_t alignedSize_ = 0;
  };

  MemoryRegionDesc getNextFreeOffset(uint32_t size);
  void flushOutstandingFences();

 private:
  VulkanContext& ctx_;
  BufferHandle stagingBuffer_;
  std::unique_ptr<lvk::VulkanImmediateCommands> immediate_;
  uint32_t stagingBufferFrontOffset_ = 0;
  uint32_t stagingBufferAlignment_ = 16;
  uint32_t stagingBufferSize_ = 0;
  uint32_t bufferCapacity_ = 0;
  std::unordered_map<uint64_t, MemoryRegionDesc> outstandingFences_;
};

} // namespace lvk
