/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>
#include <set>
#include <vector>

#define VMA_IMPLEMENTATION
#define VOLK_IMPLEMENTATION

#include <igl/vulkan/Device.h>
#include <igl/vulkan/VulkanBuffer.h>
#include <igl/vulkan/VulkanContext.h>
#include <igl/vulkan/VulkanDescriptorSetLayout.h>
#include <igl/vulkan/VulkanDevice.h>
#include <igl/vulkan/VulkanExtensions.h>
#include <igl/vulkan/VulkanPipelineBuilder.h>
#include <igl/vulkan/VulkanPipelineLayout.h>
#include <igl/vulkan/VulkanSampler.h>
#include <igl/vulkan/VulkanSemaphore.h>
#include <igl/vulkan/VulkanSwapchain.h>
#include <igl/vulkan/VulkanTexture.h>
#include <igl/vulkan/VulkanVma.h>

#if IGL_PLATFORM_MACOS
#include <dlfcn.h>
#endif

namespace {

const char* kDefaultValidationLayers[] = {"VK_LAYER_KHRONOS_validation"};

/*
 These bindings should match GLSL declarations injected into shaders in
 Device::compileShaderModule(). Same with SparkSL.
 */
const uint32_t kBinding_Texture2D = 0;
const uint32_t kBinding_Texture2DArray = 1;
const uint32_t kBinding_Texture3D = 2;
const uint32_t kBinding_TextureCube = 3;
const uint32_t kBinding_Sampler = 4;
const uint32_t kBinding_SamplerShadow = 5;
const uint32_t kBinding_StorageImages = 6;

// TODO: Implement VK_EXT_debug_report functions
#if defined(VK_EXT_debug_utils) && IGL_PLATFORM_WIN
VKAPI_ATTR VkBool32 VKAPI_CALL
vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT msgSeverity,
                    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT msgType,
                    const VkDebugUtilsMessengerCallbackDataEXT* cbData,
                    void* userData) {
  if (msgSeverity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    return VK_FALSE;
  }

  const bool isError = (msgSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;

#if IGL_DEBUG || defined(IGL_FORCE_ENABLE_LOGS)
  char errorName[128] = {};
  int object = 0;
  void* handle = nullptr;
  char typeName[128] = {};
  void* messageID = nullptr;

  if (sscanf(cbData->pMessage,
             "Validation Error : [ %127s ] Object %i: handle = %p, type = %127s | MessageID = %p",
             errorName,
             &object,
             &handle,
             typeName,
             &messageID) >= 2) {
    const char* message = strrchr(cbData->pMessage, '|') + 1;
    LLOGL(
        "%sValidation layer:\n Validation Error: %s \n Object %i: handle = %p, type = %s\n "
        "MessageID = %p \n%s \n",
        isError ? "\nERROR:\n" : "",
        errorName,
        object,
        handle,
        typeName,
        messageID,
        message);
  } else {
    LLOGL("%sValidation layer:\n%s\n", isError ? "\nERROR:\n" : "", cbData->pMessage);
  }
#endif

  if (isError) {
    igl::vulkan::VulkanContext* ctx = static_cast<igl::vulkan::VulkanContext*>(userData);
    if (ctx->config_.terminateOnValidationError) {
      IGL_ASSERT(false);
      std::terminate();
    }
  }

  return VK_FALSE;
}
#endif // defined(VK_EXT_debug_utils) && !IGL_PLATFORM_ANDROID

std::vector<VkFormat> getCompatibleDepthStencilFormats(igl::TextureFormat format) {
  switch (format) {
  case igl::TextureFormat::Z_UNorm16:
    return {VK_FORMAT_D16_UNORM,
            VK_FORMAT_D16_UNORM_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
            VK_FORMAT_D32_SFLOAT};
  case igl::TextureFormat::Z_UNorm24:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM_S8_UINT};
  case igl::TextureFormat::Z_UNorm32:
    return {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
  case igl::TextureFormat::S8_UInt_Z24_UNorm:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT};
  case igl::TextureFormat::S8_UInt_Z32_UNorm:
    return {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT};
  case igl::TextureFormat::S_UInt8:
    return {VK_FORMAT_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    // default
  default:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT};
  }
}

VkQueueFlagBits getQueueTypeFlag(igl::CommandQueueType type) {
  switch (type) {
  case igl::CommandQueueType::Compute:
    return VK_QUEUE_COMPUTE_BIT;
  case igl::CommandQueueType::Graphics:
    return VK_QUEUE_GRAPHICS_BIT;
  case igl::CommandQueueType::Transfer:
    return VK_QUEUE_TRANSFER_BIT;
  }
  IGL_UNREACHABLE_RETURN(VK_QUEUE_GRAPHICS_BIT);
}

bool validateImageLimits(VkImageType imageType,
                         VkSampleCountFlagBits samples,
                         const VkExtent3D& extent,
                         const VkPhysicalDeviceLimits& limits,
                         igl::Result* outResult) {
  using igl::Result;

  if (samples != VK_SAMPLE_COUNT_1_BIT && !IGL_VERIFY(imageType == VK_IMAGE_TYPE_2D)) {
    Result::setResult(
        outResult,
        Result(Result::Code::InvalidOperation, "Multisampling is supported only for 2D images"));
    return false;
  }

  if (imageType == VK_IMAGE_TYPE_1D && !IGL_VERIFY(extent.width <= limits.maxImageDimension1D)) {
    Result::setResult(outResult,
                      Result(Result::Code::InvalidOperation, "1D texture size exceeded"));
    return false;
  } else if (imageType == VK_IMAGE_TYPE_2D &&
             !IGL_VERIFY(extent.width <= limits.maxImageDimension2D &&
                         extent.height <= limits.maxImageDimension2D)) {
    Result::setResult(outResult,
                      Result(Result::Code::InvalidOperation, "2D texture size exceeded"));
    return false;
  } else if (imageType == VK_IMAGE_TYPE_3D &&
             !IGL_VERIFY(extent.width <= limits.maxImageDimension3D &&
                         extent.height <= limits.maxImageDimension3D &&
                         extent.depth <= limits.maxImageDimension3D)) {
    Result::setResult(outResult,
                      Result(Result::Code::InvalidOperation, "3D texture size exceeded"));
    return false;
  }

  Result::setOk(outResult);

  return true;
}

} // namespace

namespace igl {
namespace vulkan {

struct VulkanContextImpl final {
  // Vulkan Memory Allocator
  VmaAllocator vma_ = VK_NULL_HANDLE;
};

VulkanContext::VulkanContext(const VulkanContextConfig& config,
                             void* window,
                             size_t numExtraInstanceExtensions,
                             const char** extraInstanceExtensions,
                             void* display) :
  config_(config) {
  IGL_PROFILER_THREAD("MainThread");

  pimpl_ = std::make_unique<VulkanContextImpl>();

  if (volkInitialize() != VK_SUCCESS) {
    LLOGW("volkInitialize() failed\n");
    exit(255);
  };

  glslang_initialize_process();

  createInstance(numExtraInstanceExtensions, extraInstanceExtensions);

  if (window) {
    createSurface(window, display);
  }
}

VulkanContext::~VulkanContext() {
  IGL_PROFILER_FUNCTION();

  if (device_) {
    waitIdle();
  }

  textures_.clear();
  samplers_.clear();

  // This will free an internal buffer that was allocated by VMA
  stagingDevice_.reset(nullptr);

  VkDevice device = device_ ? device_->getVkDevice() : VK_NULL_HANDLE;

  for (auto r : renderPasses_) {
    vkDestroyRenderPass(device, r, nullptr);
  }
  
  dslBindless_.reset(nullptr);
  pipelineLayoutGraphics_.reset(nullptr);
  pipelineLayoutCompute_.reset(nullptr);
  swapchain_.reset(nullptr); // Swapchain has to be destroyed prior to Surface

  waitDeferredTasks();

  immediate_.reset(nullptr);

  vkDestroyDescriptorPool(device, dpBindless_, nullptr);
  vkDestroyPipelineCache(device, pipelineCache_, nullptr);
  vkDestroySurfaceKHR(vkInstance_, vkSurface_, nullptr);

  // Clean up VMA
  if (IGL_VULKAN_USE_VMA) {
    vmaDestroyAllocator(pimpl_->vma_);
  }

  device_.reset(nullptr); // Device has to be destroyed prior to Instance
#if defined(VK_EXT_debug_utils) && !IGL_PLATFORM_ANDROID
  vkDestroyDebugUtilsMessengerEXT(vkInstance_, vkDebugUtilsMessenger_, nullptr);
#endif // defined(VK_EXT_debug_utils) && !IGL_PLATFORM_ANDROID
  vkDestroyInstance(vkInstance_, nullptr);

  glslang_finalize_process();

#if IGL_DEBUG || defined(IGL_FORCE_ENABLE_LOGS)
  LLOGL("Vulkan graphics pipelines created: %u\n",
               VulkanPipelineBuilder::getNumPipelinesCreated());
  LLOGL("Vulkan compute pipelines created: %u\n",
               VulkanComputePipelineBuilder::getNumPipelinesCreated());
#endif // IGL_DEBUG || defined(IGL_FORCE_ENABLE_LOGS)
}

void VulkanContext::createInstance(const size_t numExtraExtensions, const char** extraExtensions) {
  // Enumerate all instance extensions
  extensions_.enumerate();
  extensions_.enableCommonExtensions(VulkanExtensions::ExtensionType::Instance,
                                     config_.enableValidation);
  for (size_t index = 0; index < numExtraExtensions; ++index) {
    extensions_.enable(extraExtensions[index], VulkanExtensions::ExtensionType::Instance);
  }

  auto instanceExtensions = extensions_.allEnabled(VulkanExtensions::ExtensionType::Instance);

  vkInstance_ = VK_NULL_HANDLE;
  // Validation Features not available on most Android devices
#if !IGL_PLATFORM_ANDROID
  const VkValidationFeatureEnableEXT validationFeaturesEnabled[] = {
      VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
  };

  const VkValidationFeaturesEXT features = {
      .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
      .pNext = nullptr,
      .enabledValidationFeatureCount =
          config_.enableGPUAssistedValidation ? IGL_ARRAY_NUM_ELEMENTS(validationFeaturesEnabled) : 0,
      .pEnabledValidationFeatures = config_.enableGPUAssistedValidation ? validationFeaturesEnabled : nullptr,
  };
#endif // !IGL_PLATFORM_ANDROID

  const VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = "IGL/Vulkan",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "IGL/Vulkan",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_3,
  };

  const VkInstanceCreateInfo ci = {
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pNext = config_.enableValidation ? &features : nullptr,
    .flags = 0,
    .pApplicationInfo = &appInfo,
    .enabledLayerCount = config_.enableValidation ? IGL_ARRAY_NUM_ELEMENTS(kDefaultValidationLayers) : 0,
    .ppEnabledLayerNames = config_.enableValidation ? kDefaultValidationLayers : nullptr,
    .enabledExtensionCount = (uint32_t)instanceExtensions.size(),
    .ppEnabledExtensionNames = instanceExtensions.data(),
  };

  VK_ASSERT(vkCreateInstance(&ci, NULL, &vkInstance_));

  volkLoadInstance(vkInstance_);

#if defined(VK_EXT_debug_utils) && IGL_PLATFORM_WIN
  if (extensions_.enabled(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
    VK_ASSERT(ivkCreateDebugUtilsMessenger(
        vkInstance_, &vulkanDebugCallback, this, &vkDebugUtilsMessenger_));
  }
#endif // if defined(VK_EXT_debug_utils) && IGL_PLATFORM_WIN

#if IGL_DEBUG || defined(IGL_FORCE_ENABLE_LOGS)
  // log available instance extensions
  LLOGL("Vulkan instance extensions:\n");
  for (const auto& extension :
       extensions_.allAvailableExtensions(VulkanExtensions::ExtensionType::Instance)) {
    LLOGL("  %s\n", extension.c_str());
  }
#endif
}

void VulkanContext::createSurface(void* window, void* display) {
  VK_ASSERT(ivkCreateSurface(vkInstance_, window, display, &vkSurface_));
}

igl::Result VulkanContext::queryDevices(const HWDeviceQueryDesc& desc,
                                        std::vector<HWDeviceDesc>& outDevices) {
  outDevices.clear();

  // Physical devices
  uint32_t deviceCount = 0;
  VK_ASSERT_RETURN(vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, nullptr));
  std::vector<VkPhysicalDevice> vkDevices(deviceCount);
  VK_ASSERT_RETURN(vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, vkDevices.data()));

  auto convertVulkanDeviceTypeToIGL = [](VkPhysicalDeviceType vkDeviceType) -> HWDeviceType {
    switch (vkDeviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return HWDeviceType::IntegratedGpu;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return HWDeviceType::DiscreteGpu;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return HWDeviceType::ExternalGpu;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return HWDeviceType::SoftwareGpu;
    default:
      return HWDeviceType::Unknown;
    }
  };

  const HWDeviceType desiredDeviceType = desc.hardwareType;

  for (uint32_t i = 0; i < deviceCount; ++i) {
    VkPhysicalDevice physicalDevice = vkDevices[i];
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

    const HWDeviceType deviceType = convertVulkanDeviceTypeToIGL(deviceProperties.deviceType);

    // filter non-suitable hardware devices
    if (desiredDeviceType != HWDeviceType::Unknown && deviceType != desiredDeviceType) {
      continue;
    }

    outDevices.emplace_back((uintptr_t)vkDevices[i],
                            deviceType,
                            deviceProperties.deviceName,
                            std::to_string(deviceProperties.vendorID));
  }

  if (outDevices.empty()) {
    return Result(Result::Code::Unsupported, "No Vulkan devices matching your criteria");
  }

  return Result();
}

igl::Result VulkanContext::initContext(const HWDeviceDesc& desc,
                                       size_t numExtraDeviceExtensions,
                                       const char** extraDeviceExtensions) {
  if (desc.guid == 0UL) {
    LLOGW("Invalid hardwareGuid(%lu)", desc.guid);
    return Result(Result::Code::Unsupported, "Vulkan is not supported");
  }

  vkPhysicalDevice_ = (VkPhysicalDevice)desc.guid;

  useStaging_ = !ivkIsHostVisibleSingleHeapMemory(vkPhysicalDevice_);

  vkGetPhysicalDeviceFeatures2(vkPhysicalDevice_, &vkPhysicalDeviceFeatures2_);
  vkGetPhysicalDeviceProperties2(vkPhysicalDevice_, &vkPhysicalDeviceProperties2_);

  const uint32_t apiVersion = vkPhysicalDeviceProperties2_.properties.apiVersion;

  LLOGL("Vulkan physical device: %s\n", vkPhysicalDeviceProperties2_.properties.deviceName);
  LLOGL("           API version: %i.%i.%i.%i\n",
               VK_API_VERSION_MAJOR(apiVersion),
               VK_API_VERSION_MINOR(apiVersion),
               VK_API_VERSION_PATCH(apiVersion),
               VK_API_VERSION_VARIANT(apiVersion));
  LLOGL("           Driver info: %s %s\n",
               vkPhysicalDeviceDriverProperties_.driverName,
               vkPhysicalDeviceDriverProperties_.driverInfo);

  extensions_.enumerate(vkPhysicalDevice_);

  LLOGL("Vulkan physical device extensions:\n");

#if IGL_DEBUG || defined(IGL_FORCE_ENABLE_LOGS)
  // log available physical device extensions
  for (const auto& extension :
       extensions_.allAvailableExtensions(VulkanExtensions::ExtensionType::Device)) {
    LLOGL("  %s\n", extension.c_str());
  }
#endif

  extensions_.enableCommonExtensions(VulkanExtensions::ExtensionType::Device);
  // Enable extra device extensions
  for (size_t i = 0; i < numExtraDeviceExtensions; i++) {
    extensions_.enable(extraDeviceExtensions[i], VulkanExtensions::ExtensionType::Device);
  }

  VulkanQueuePool queuePool(vkPhysicalDevice_);

  // Reserve IGL Vulkan queues
  auto graphicsQueueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_GRAPHICS_BIT);
  auto computeQueueDescriptor = queuePool.findQueueDescriptor(VK_QUEUE_COMPUTE_BIT);

  if (!graphicsQueueDescriptor.isValid()) {
    LLOGW("VK_QUEUE_GRAPHICS_BIT is not supported");
    return Result(Result::Code::Unsupported, "VK_QUEUE_GRAPHICS_BIT is not supported");
  }

  if (!computeQueueDescriptor.isValid()) {
    LLOGW("VK_QUEUE_COMPUTE_BIT is not supported");
    return Result(Result::Code::Unsupported, "VK_QUEUE_COMPUTE_BIT is not supported");
  }

  deviceQueues_.graphicsQueueFamilyIndex = graphicsQueueDescriptor.familyIndex;
  deviceQueues_.computeQueueFamilyIndex = computeQueueDescriptor.familyIndex;

  queuePool.reserveQueue(graphicsQueueDescriptor);
  queuePool.reserveQueue(computeQueueDescriptor);

  // Reserve queues requested by user
  // Reserve transfer types at the end, since those can fallback to compute and graphics queues.
  // This reduces the risk of failing reservation due to saturation of compute and graphics queues
  auto sortedUserQueues = config_.userQueues;
  sort(sortedUserQueues.begin(), sortedUserQueues.end(), [](const auto& /*q1*/, const auto& q2) {
    return q2 == CommandQueueType::Transfer;
  });

  for (const auto& userQueue : sortedUserQueues) {
    auto userQueueDescriptor = queuePool.findQueueDescriptor(getQueueTypeFlag(userQueue));
    if (userQueueDescriptor.isValid()) {
      userQueues_[userQueue] = userQueueDescriptor;
    } else {
      LLOGW("User requested queue is not supported");
      return Result(Result::Code::Unsupported, "User requested queue is not supported");
    }
  }

  for (const auto& [_, descriptor] : userQueues_) {
    queuePool.reserveQueue(descriptor);
  }

  const auto qcis = queuePool.getQueueCreationInfos();

  assert(qcis.size() >= 1);

  VkPhysicalDeviceFeatures deviceFeatures = {
      .multiDrawIndirect = VK_TRUE,
      .drawIndirectFirstInstance = VK_TRUE,
      .depthBiasClamp = VK_TRUE,
      .fillModeNonSolid = VK_TRUE,
      .shaderInt16 = VK_TRUE,
  };
  VkPhysicalDeviceVulkan11Features deviceFeatures11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .storageBuffer16BitAccess = VK_TRUE,
      .shaderDrawParameters = VK_TRUE,
  };
  VkPhysicalDeviceVulkan12Features deviceFeatures12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &deviceFeatures11,
      .shaderFloat16 = VK_TRUE,
      .descriptorIndexing = VK_TRUE,
      .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
      .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
      .descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
      .descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
      .descriptorBindingPartiallyBound = VK_TRUE,
      .descriptorBindingVariableDescriptorCount = VK_TRUE,
      .runtimeDescriptorArray = VK_TRUE,
      .uniformBufferStandardLayout = VK_TRUE,
      .timelineSemaphore = VK_TRUE,
      .bufferDeviceAddress = VK_TRUE,
      .bufferDeviceAddressCaptureReplay = VK_TRUE,
  };
  VkPhysicalDeviceVulkan13Features deviceFeatures13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &deviceFeatures12,
      .subgroupSizeControl = VK_TRUE,
      .synchronization2 = VK_TRUE,
      .dynamicRendering = VK_TRUE,
      .maintenance4 = VK_TRUE,
  };
  const VkDeviceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &deviceFeatures13,
      .queueCreateInfoCount = (uint32_t)qcis.size(),
      .pQueueCreateInfos = qcis.data(),
      .enabledLayerCount = (uint32_t)IGL_ARRAY_NUM_ELEMENTS(kDefaultValidationLayers),
      .ppEnabledLayerNames = kDefaultValidationLayers,
      .enabledExtensionCount =
          (uint32_t)extensions_.allEnabled(VulkanExtensions::ExtensionType::Device).size(),
      .ppEnabledExtensionNames =
          extensions_.allEnabled(VulkanExtensions::ExtensionType::Device).data(),
      .pEnabledFeatures = &deviceFeatures,
  };

  VkDevice device;
  VK_ASSERT_RETURN(vkCreateDevice(vkPhysicalDevice_, &ci, NULL, &device));

  volkLoadDevice(device);

  vkGetDeviceQueue(device, deviceQueues_.graphicsQueueFamilyIndex, 0, &deviceQueues_.graphicsQueue);
  vkGetDeviceQueue(device, deviceQueues_.computeQueueFamilyIndex, 0, &deviceQueues_.computeQueue);

  device_ = std::make_unique<igl::vulkan::VulkanDevice>(device, "Device: VulkanContext::device_");
  immediate_ = std::make_unique<igl::vulkan::VulkanImmediateCommands>(
      device, deviceQueues_.graphicsQueueFamilyIndex, "VulkanContext::immediate_");

  // create Vulkan pipeline cache
  {
    const VkPipelineCacheCreateInfo ci = {
        VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        nullptr,
        VkPipelineCacheCreateFlags(0),
        config_.pipelineCacheDataSize,
        config_.pipelineCacheData,
    };
    vkCreatePipelineCache(device, &ci, nullptr, &pipelineCache_);
  }

  // Create Vulkan Memory Allocator
  if (IGL_VULKAN_USE_VMA) {
    VK_ASSERT_RETURN(ivkVmaCreateAllocator(
        vkPhysicalDevice_, device_->getVkDevice(), vkInstance_, apiVersion, &pimpl_->vma_));
  }

  // The staging device will use VMA to allocate a buffer, so this needs
  // to happen after VMA has been initialized.
  stagingDevice_ = std::make_unique<igl::vulkan::VulkanStagingDevice>(*this);

  // default texture
  IGL_ASSERT(textures_.size() == 1);
  {
    const VkFormat dummyTextureFormat = VK_FORMAT_R8G8B8A8_UNORM;
    const VkMemoryPropertyFlags memFlags = useStaging_ ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                                       : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    Result result;
    auto image = createImage(VK_IMAGE_TYPE_2D,
                             VkExtent3D{1, 1, 1},
                             dummyTextureFormat,
                             1,
                             1,
                             VK_IMAGE_TILING_OPTIMAL,
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_STORAGE_BIT,
                             memFlags,
                             0,
                             VK_SAMPLE_COUNT_1_BIT,
                             &result,
                             "Image: dummy 1x1");
    if (!IGL_VERIFY(result.isOk())) {
      return result;
    }
    if (!IGL_VERIFY(image)) {
      return Result(Result::Code::InvalidOperation, "Cannot create VulkanImage");
    }
    auto imageView = image->createImageView(VK_IMAGE_VIEW_TYPE_2D,
                                            dummyTextureFormat,
                                            VK_IMAGE_ASPECT_COLOR_BIT,
                                            0,
                                            VK_REMAINING_MIP_LEVELS,
                                            0,
                                            1,
                                            "Image View: dummy 1x1");
    if (!IGL_VERIFY(imageView)) {
      return Result(Result::Code::InvalidOperation, "Cannot create VulkanImageView");
    }
    textures_[0] = std::make_shared<VulkanTexture>(*this, std::move(image), std::move(imageView));
    const uint32_t pixel = 0xFF000000;
    const VkRect2D imageRegion = ivkGetRect2D(0, 0, 1, 1);
    stagingDevice_->imageData2D(
        textures_[0]->getVulkanImage(), imageRegion, 0, 1, 0, dummyTextureFormat, &pixel);
  }

  // default sampler
  IGL_ASSERT(samplers_.size() == 1);
  samplers_[0] =
      std::make_shared<VulkanSampler>(*this,
                                      device,
                                      ivkGetSamplerCreateInfo(VK_FILTER_LINEAR,
                                                              VK_FILTER_LINEAR,
                                                              VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                                              VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                              VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                              VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                              0.0f,
                                                              0.0f),
                                      "Sampler: default");

  if (!IGL_VERIFY(
          config_.maxSamplers <=
          vkPhysicalDeviceDescriptorIndexingProperties_.maxDescriptorSetUpdateAfterBindSamplers)) {
    LLOGW(
        "Max Samplers exceeded %u (max %u)",
        config_.maxSamplers,
        vkPhysicalDeviceDescriptorIndexingProperties_.maxDescriptorSetUpdateAfterBindSamplers);
  }

  if (!IGL_VERIFY(config_.maxTextures <= vkPhysicalDeviceDescriptorIndexingProperties_
                                             .maxDescriptorSetUpdateAfterBindSampledImages)) {
    LLOGW(
        "Max Textures exceeded: %u (max %u)",
        config_.maxTextures,
        vkPhysicalDeviceDescriptorIndexingProperties_.maxDescriptorSetUpdateAfterBindSampledImages);
  }

  const VkPhysicalDeviceLimits& limits = getVkPhysicalDeviceProperties().limits;

  {
    // create default descriptor set layout which is going to be shared by graphics pipelines
    constexpr uint32_t numBindings = 7;
    const VkDescriptorSetLayoutBinding bindings[numBindings] = {
        ivkGetDescriptorSetLayoutBinding(
            kBinding_Texture2D, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, config_.maxTextures),
        ivkGetDescriptorSetLayoutBinding(
            kBinding_Texture2DArray, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, config_.maxTextures),
        ivkGetDescriptorSetLayoutBinding(
            kBinding_Texture3D, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, config_.maxTextures),
        ivkGetDescriptorSetLayoutBinding(
            kBinding_TextureCube, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, config_.maxTextures),
        ivkGetDescriptorSetLayoutBinding(
            kBinding_Sampler, VK_DESCRIPTOR_TYPE_SAMPLER, config_.maxSamplers),
        ivkGetDescriptorSetLayoutBinding(
            kBinding_SamplerShadow, VK_DESCRIPTOR_TYPE_SAMPLER, config_.maxSamplers),
        ivkGetDescriptorSetLayoutBinding(
            kBinding_StorageImages, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, config_.maxTextures),
    };
    const uint32_t flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                           VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    const VkDescriptorBindingFlags bindingFlags[numBindings] = {
        flags, flags, flags, flags, flags, flags, flags};
    dslBindless_ = std::make_unique<VulkanDescriptorSetLayout>(
        device,
        numBindings,
        bindings,
        bindingFlags,
        "Descriptor Set Layout: VulkanContext::dslBindless_");

    // create default descriptor pool and allocate 1 descriptor set
    const uint32_t numSets = 1;
    IGL_ASSERT(numSets > 0);
    const VkDescriptorPoolSize poolSizes[numBindings] {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, numSets * config_.maxTextures},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, numSets * config_.maxTextures},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, numSets * config_.maxTextures},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, numSets * config_.maxTextures},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, numSets * config_.maxSamplers},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, numSets * config_.maxSamplers},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, numSets * config_.maxTextures},
    };
    bindlessDSets_.resize(numSets);
    VK_ASSERT_RETURN(ivkCreateDescriptorPool(
        device, numSets, numBindings, poolSizes, &dpBindless_));
    for (size_t i = 0; i != numSets; i++) {
      VK_ASSERT_RETURN(ivkAllocateDescriptorSet(
          device, dpBindless_, dslBindless_->getVkDescriptorSetLayout(), &bindlessDSets_[i].ds));
    }
  }

  // maxPushConstantsSize is guaranteed to be at least 128 bytes
  // https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#features-limits
  // Table 32. Required Limits
  const uint32_t kPushConstantsSize = 128;
  if (!IGL_VERIFY(kPushConstantsSize <= limits.maxPushConstantsSize)) {
    LLOGW("Push constants size exceeded %u (max %u bytes)",
                  kPushConstantsSize,
                  limits.maxPushConstantsSize);
  }

  VkDescriptorSetLayout dsl = dslBindless_->getVkDescriptorSetLayout();

  // create pipeline layout
  pipelineLayoutGraphics_ = std::make_unique<VulkanPipelineLayout>(
      device,
      dsl,
      ivkGetPushConstantRange(
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, kPushConstantsSize),
      "Pipeline Layout: VulkanContext::pipelineLayoutGraphics_");

  pipelineLayoutCompute_ = std::make_unique<VulkanPipelineLayout>(
      device,
      dsl,
      ivkGetPushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushConstantsSize),
      "Pipeline Layout: VulkanContext::pipelineLayoutCompute_");

  querySurfaceCapabilities();

  return Result();
}

igl::Result VulkanContext::initSwapchain(uint32_t width, uint32_t height) {
  if (!device_ || !immediate_) {
    LLOGW("Call initContext() first");
    return Result(Result::Code::Unsupported, "Call initContext() first");
  }

  if (swapchain_) {
    vkDeviceWaitIdle(device_->device_);
    swapchain_ = nullptr; // Destroy old swapchain first
  }

  swapchain_ = std::make_unique<igl::vulkan::VulkanSwapchain>(*this, width, height);

  return swapchain_ ? Result() : Result(Result::Code::RuntimeError, "Failed to create Swapchain");
}

VkExtent2D VulkanContext::getSwapchainExtent() const {
  return hasSwapchain() ? swapchain_->getExtent() : VkExtent2D{0, 0};
}

Result VulkanContext::waitIdle() const {
  IGL_PROFILER_FUNCTION_COLOR(IGL_PROFILER_COLOR_WAIT);

  for (auto queue : {deviceQueues_.graphicsQueue, deviceQueues_.computeQueue}) {
    VK_ASSERT_RETURN(vkQueueWaitIdle(queue));
  }

  return getResultFromVkResult(VK_SUCCESS);
}

Result VulkanContext::present() const {
  if (!hasSwapchain()) {
    return Result(Result::Code::InvalidOperation, "No swapchain available");
  }

  return swapchain_->present(immediate_->acquireLastSubmitSemaphore());
}

std::shared_ptr<VulkanBuffer> VulkanContext::createBuffer(VkDeviceSize bufferSize,
                                                          VkBufferUsageFlags usageFlags,
                                                          VkMemoryPropertyFlags memFlags,
                                                          igl::Result* outResult,
                                                          const char* debugName) const {
#define ENSURE_BUFFER_SIZE(flag, maxSize)                                                      \
  if (usageFlags & flag) {                                                                     \
    if (!IGL_VERIFY(bufferSize <= maxSize)) {                                                  \
      Result::setResult(outResult,                                                             \
                        Result(Result::Code::InvalidOperation, "Buffer size exceeded" #flag)); \
      return nullptr;                                                                          \
    }                                                                                          \
  }

  const VkPhysicalDeviceLimits& limits = getVkPhysicalDeviceProperties().limits;

  ENSURE_BUFFER_SIZE(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, limits.maxUniformBufferRange);
  // any buffer
  ENSURE_BUFFER_SIZE(VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM, limits.maxStorageBufferRange);
#undef ENSURE_BUFFER_SIZE

  Result::setOk(outResult);
  return std::make_shared<VulkanBuffer>(
      *this, device_->getVkDevice(), bufferSize, usageFlags, memFlags, debugName);
}

std::shared_ptr<VulkanImage> VulkanContext::createImage(VkImageType imageType,
                                                        VkExtent3D extent,
                                                        VkFormat format,
                                                        uint32_t mipLevels,
                                                        uint32_t arrayLayers,
                                                        VkImageTiling tiling,
                                                        VkImageUsageFlags usageFlags,
                                                        VkMemoryPropertyFlags memFlags,
                                                        VkImageCreateFlags flags,
                                                        VkSampleCountFlagBits samples,
                                                        igl::Result* outResult,
                                                        const char* debugName) const {
  if (!validateImageLimits(
          imageType, samples, extent, getVkPhysicalDeviceProperties().limits, outResult)) {
    return nullptr;
  }

  return std::make_shared<VulkanImage>(*this,
                                       device_->getVkDevice(),
                                       extent,
                                       imageType,
                                       format,
                                       mipLevels,
                                       arrayLayers,
                                       tiling,
                                       usageFlags,
                                       memFlags,
                                       flags,
                                       samples,
                                       debugName);
}

std::shared_ptr<VulkanImage> VulkanContext::createImageFromFileDescriptor(
    int32_t fileDescriptor,
    uint64_t memoryAllocationSize,
    VkImageType imageType,
    VkExtent3D extent,
    VkFormat format,
    uint32_t mipLevels,
    uint32_t arrayLayers,
    VkImageTiling tiling,
    VkImageUsageFlags usageFlags,
    VkImageCreateFlags flags,
    VkSampleCountFlagBits samples,
    igl::Result* outResult,
    const char* debugName) const {
  if (!validateImageLimits(
          imageType, samples, extent, getVkPhysicalDeviceProperties().limits, outResult)) {
    return nullptr;
  }

  return std::make_shared<VulkanImage>(*this,
                                       fileDescriptor,
                                       memoryAllocationSize,
                                       device_->getVkDevice(),
                                       extent,
                                       imageType,
                                       format,
                                       mipLevels,
                                       arrayLayers,
                                       tiling,
                                       usageFlags,
                                       flags,
                                       samples,
                                       debugName);
}

void VulkanContext::bindDefaultDescriptorSets(VkCommandBuffer cmdBuf,
                                              VkPipelineBindPoint bindPoint) const {
  IGL_PROFILER_FUNCTION();

  const bool isGraphics = bindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS;

  const VkDescriptorSet sets[] = {
      bindlessDSets_[currentDSetIndex_].ds,
  };
#if IGL_DEBUG_DESCRIPTOR_SETS
  IGL_LOG_INFO("Binding descriptor set %u\n", currentDSetIndex_);
#endif // IGL_DEBUG_DESCRIPTOR_SETS
  vkCmdBindDescriptorSets(
      cmdBuf,
      bindPoint,
      (isGraphics ? pipelineLayoutGraphics_ : pipelineLayoutCompute_)->getVkPipelineLayout(),
      0,
      IGL_ARRAY_NUM_ELEMENTS(sets),
      sets,
      0,
      nullptr);
}

void VulkanContext::checkAndUpdateDescriptorSets() const {
  if (awaitingDeletion_) {
    // Our descriptor set was created with VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT which
    // indicates that descriptors in this binding that are not dynamically used need not contain
    // valid descriptors at the time the descriptors are consumed. A descriptor is dynamically used
    // if any shader invocation executes an instruction that performs any memory access using the
    // descriptor. If a descriptor is not dynamically used, any resource referenced by the
    // descriptor is not considered to be referenced during command execution.
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorBindingFlagBits.html
    awaitingDeletion_ = false;
  }

  if (!awaitingCreation_) {
    // nothing to update here
    return;
  }

  // newly created resources can be used immediately - make sure they are put into descriptor sets
  IGL_PROFILER_FUNCTION();

  // here we remove deleted textures - everything which has only 1 reference is owned by this
  // context and can be released safely
  for (uint32_t i = 1; i < (uint32_t)textures_.size(); i++) {
    if (textures_[i] && textures_[i].use_count() == 1) {
      if (i == textures_.size() - 1) {
        textures_.pop_back();
      } else {
        textures_[i].reset();
        freeIndicesTextures_.push_back(i);
      }
    }
  }
  for (uint32_t i = 1; i < (uint32_t)samplers_.size(); i++) {
    if (samplers_[i] && samplers_[i].use_count() == 1) {
      if (i == samplers_.size() - 1) {
        samplers_.pop_back();
      } else {
        samplers_[i].reset();
        freeIndicesSamplers_.push_back(i);
      }
    }
  }

  // update Vulkan descriptor set here

  // 1. Sampled and storage images
  std::vector<VkDescriptorImageInfo> infoSampledImages;
  std::vector<VkDescriptorImageInfo> infoStorageImages;
  IGL_ASSERT(textures_.size() >= 1); // make sure the guard value is always there
  infoSampledImages.reserve(textures_.size());
  infoStorageImages.reserve(textures_.size());

  // use the dummy texture to avoid sparse array
  VkImageView dummyImageView = textures_[0]->imageView_->getVkImageView();

  for (const auto& texture : textures_) {
    // multisampled images cannot be directly accessed from shaders
    const bool isTextureAvailable =
        texture && ((texture->image_->samples_ & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT);
    const bool isSampledImage = isTextureAvailable && texture->image_->isSampledImage();
    const bool isStorageImage = isTextureAvailable && texture->image_->isStorageImage();
    infoSampledImages.push_back(
        {samplers_[0]->getVkSampler(),
         isSampledImage ? texture->imageView_->getVkImageView() : dummyImageView,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    IGL_ASSERT(infoSampledImages.back().imageView != VK_NULL_HANDLE);
    infoStorageImages.push_back(VkDescriptorImageInfo{
        VK_NULL_HANDLE,
        isStorageImage ? texture->imageView_->getVkImageView() : dummyImageView,
        VK_IMAGE_LAYOUT_GENERAL});
  }

  // 2. Samplers
  std::vector<VkDescriptorImageInfo> infoSamplers;
  IGL_ASSERT(samplers_.size() >= 1); // make sure the guard value is always there
  infoSamplers.reserve(samplers_.size());

  for (const auto& sampler : samplers_) {
    infoSamplers.push_back({(sampler ? sampler : samplers_[0])->getVkSampler(),
                            VK_NULL_HANDLE,
                            VK_IMAGE_LAYOUT_UNDEFINED});
  }

  std::vector<VkWriteDescriptorSet> write;

  // we want to update the next available descriptor set
  const uint32_t nextDSetIndex = (currentDSetIndex_ + 1) % bindlessDSets_.size();
  auto& dsetToUpdate = bindlessDSets_[nextDSetIndex];

  if (!infoSampledImages.empty()) {
    // use the same indexing for every texture type
    for (uint32_t i = kBinding_Texture2D; i != kBinding_TextureCube + 1; i++) {
      write.push_back(ivkGetWriteDescriptorSet_ImageInfo(dsetToUpdate.ds,
                                                         i,
                                                         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                                         (uint32_t)infoSampledImages.size(),
                                                         infoSampledImages.data()));
    }
  };

  if (!infoSamplers.empty()) {
    for (uint32_t i = kBinding_Sampler; i != kBinding_SamplerShadow + 1; i++) {
      write.push_back(ivkGetWriteDescriptorSet_ImageInfo(dsetToUpdate.ds,
                                                         i,
                                                         VK_DESCRIPTOR_TYPE_SAMPLER,
                                                         (uint32_t)infoSamplers.size(),
                                                         infoSamplers.data()));
    }
  }

  if (!infoStorageImages.empty()) {
    write.push_back(ivkGetWriteDescriptorSet_ImageInfo(dsetToUpdate.ds,
                                                       kBinding_StorageImages,
                                                       VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                       (uint32_t)infoStorageImages.size(),
                                                       infoStorageImages.data()));
  };

  // do not switch to the next descriptor set if there is nothing to update
  if (!write.empty()) {
#if IGL_VULKAN_PRINT_COMMANDS
    LLOGL("Updating descriptor set %u\n", nextDSetIndex);
#endif // IGL_VULKAN_PRINT_COMMANDS
    currentDSetIndex_ = nextDSetIndex;
    immediate_->wait(std::exchange(dsetToUpdate.handle, immediate_->getLastSubmitHandle()));
    vkUpdateDescriptorSets(
        device_->getVkDevice(), static_cast<uint32_t>(write.size()), write.data(), 0, nullptr);
  }

  awaitingCreation_ = false;
  awaitingDeletion_ = false;

  lastDeletionFrame_ = getFrameNumber();
}

std::shared_ptr<VulkanTexture> VulkanContext::createTexture(
    std::shared_ptr<VulkanImage> image,
    std::shared_ptr<VulkanImageView> imageView) const {
  auto texture = std::make_shared<VulkanTexture>(*this, std::move(image), std::move(imageView));
  if (!IGL_VERIFY(texture)) {
    return nullptr;
  }
  if (!freeIndicesTextures_.empty()) {
    // reuse an empty slot
    texture->textureId_ = freeIndicesTextures_.back();
    freeIndicesTextures_.pop_back();
    textures_[texture->textureId_] = texture;
  } else {
    texture->textureId_ = uint32_t(textures_.size());
    textures_.emplace_back(texture);
  }

  IGL_ASSERT(textures_.size() <= config_.maxTextures);

  awaitingCreation_ = true;

  return texture;
}

std::shared_ptr<VulkanSampler> VulkanContext::createSampler(const VkSamplerCreateInfo& ci,
                                                            igl::Result* outResult,
                                                            const char* debugName) const {
  auto sampler = std::make_shared<VulkanSampler>(*this, device_->getVkDevice(), ci, debugName);
  if (!IGL_VERIFY(sampler)) {
    Result::setResult(outResult, Result::Code::InvalidOperation);
    return nullptr;
  }
  if (!freeIndicesSamplers_.empty()) {
    // reuse an empty slot
    sampler->samplerId_ = freeIndicesSamplers_.back();
    freeIndicesSamplers_.pop_back();
    samplers_[sampler->samplerId_] = sampler;
  } else {
    sampler->samplerId_ = uint32_t(samplers_.size());
    samplers_.emplace_back(sampler);
  }

  IGL_ASSERT(samplers_.size() <= config_.maxSamplers);

  awaitingCreation_ = true;

  return sampler;
}

void VulkanContext::querySurfaceCapabilities() {
  // This is not an exhaustive list. It's only formats that we are using.
  std::vector<VkFormat> depthFormats = {VK_FORMAT_D32_SFLOAT_S8_UINT,
                                        VK_FORMAT_D24_UNORM_S8_UINT,
                                        VK_FORMAT_D16_UNORM_S8_UINT,
                                        VK_FORMAT_D32_SFLOAT,
                                        VK_FORMAT_D16_UNORM,
                                        VK_FORMAT_S8_UINT};
  for (const auto& depthFormat : depthFormats) {
    VkFormatProperties formatProps;
    vkGetPhysicalDeviceFormatProperties(vkPhysicalDevice_, depthFormat, &formatProps);

    if (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT ||
        formatProps.bufferFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT ||
        formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      deviceDepthFormats_.push_back(depthFormat);
    }
  }

  if (vkSurface_ != VK_NULL_HANDLE) {
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkPhysicalDevice_, vkSurface_, &deviceSurfaceCaps_);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_, vkSurface_, &formatCount, nullptr);

    if (formatCount) {
      deviceSurfaceFormats_.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(
          vkPhysicalDevice_, vkSurface_, &formatCount, deviceSurfaceFormats_.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        vkPhysicalDevice_, vkSurface_, &presentModeCount, nullptr);

    if (presentModeCount) {
      devicePresentModes_.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          vkPhysicalDevice_, vkSurface_, &presentModeCount, devicePresentModes_.data());
    }
  }
}

VkFormat VulkanContext::getClosestDepthStencilFormat(igl::TextureFormat desiredFormat) const {
  // get a list of compatible depth formats for a given desired format
  // The list will contain depth format that are ordered from most to least closest
  const std::vector<VkFormat> compatibleDepthStencilFormatList =
      getCompatibleDepthStencilFormats(desiredFormat);

  // Generate a set of device supported formats
  std::set<VkFormat> availableFormats;
  for (auto format : deviceDepthFormats_) {
    availableFormats.insert(format);
  }

  // check if any of the format in compatible list is supported
  for (auto depthStencilFormat : compatibleDepthStencilFormatList) {
    if (availableFormats.count(depthStencilFormat) != 0) {
      return depthStencilFormat;
    }
  }

  // no matching found, choose the first supported format
  return !deviceDepthFormats_.empty() ? deviceDepthFormats_[0] : VK_FORMAT_D24_UNORM_S8_UINT;
}

VulkanContext::RenderPassHandle VulkanContext::getRenderPass(uint8_t index) const {
  return RenderPassHandle{renderPasses_[index], index};
}

VulkanContext::RenderPassHandle VulkanContext::findRenderPass(
    const VulkanRenderPassBuilder& builder) const {
  IGL_PROFILER_FUNCTION();

  auto it = renderPassesHash_.find(builder);

  if (it != renderPassesHash_.end()) {
    return RenderPassHandle{renderPasses_[it->second], it->second};
  }

  VkRenderPass pass = VK_NULL_HANDLE;
  builder.build(device_->getVkDevice(), &pass);

  const size_t index = renderPasses_.size();

  IGL_ASSERT(index <= 255);

  renderPassesHash_[builder] = uint8_t(index);
  renderPasses_.push_back(pass);

  return RenderPassHandle{pass, uint8_t(index)};
}

std::vector<uint8_t> VulkanContext::getPipelineCacheData() const {
  VkDevice device = device_->getVkDevice();

  size_t size = 0;
  vkGetPipelineCacheData(device, pipelineCache_, &size, nullptr);

  std::vector<uint8_t> data(size);

  if (size) {
    vkGetPipelineCacheData(device, pipelineCache_, &size, data.data());
  }

  return data;
}

uint64_t VulkanContext::getFrameNumber() const {
  return swapchain_ ? swapchain_->getFrameNumber() : 0u;
}

void VulkanContext::deferredTask(std::packaged_task<void()>&& task, SubmitHandle handle) const {
  if (handle.empty()) {
    handle = immediate_->getLastSubmitHandle();
  }
  deferredTasks_.emplace_back(std::move(task), handle);
}

bool VulkanContext::areValidationLayersEnabled() const {
  return config_.enableValidation;
}

void* VulkanContext::getVmaAllocator() const {
  return pimpl_->vma_;
}

void VulkanContext::processDeferredTasks() const {
  while (!deferredTasks_.empty() && immediate_->isReady(deferredTasks_.front().handle_, true)) {
    deferredTasks_.front().task_();
    deferredTasks_.pop_front();
  }
}

void VulkanContext::waitDeferredTasks() {
  for (auto& task : deferredTasks_) {
    immediate_->wait(task.handle_);
    task.task_();
  }
  deferredTasks_.clear();
}

} // namespace vulkan
} // namespace igl
