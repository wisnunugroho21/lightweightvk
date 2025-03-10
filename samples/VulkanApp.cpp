/*
* LightweightVK
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

// Based on https://github.com/PacktPublishing/3D-Graphics-Rendering-Cookbook-Second-Edition/blob/main/shared/VulkanApp.cpp

#include "VulkanApp.h"

#include <filesystem>
#include <vector>

#if defined(ANDROID)
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>

static double glfwGetTime() {
  timespec t = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1.0e-9 * t.tv_nsec;
}

static const char* cmdToString(int32_t cmd) {
#define CMD(cmd) \
  case cmd:      \
    return #cmd
  switch (cmd) {
    CMD(APP_CMD_INPUT_CHANGED);
    CMD(APP_CMD_INIT_WINDOW);
    CMD(APP_CMD_TERM_WINDOW);
    CMD(APP_CMD_WINDOW_RESIZED);
    CMD(APP_CMD_WINDOW_REDRAW_NEEDED);
    CMD(APP_CMD_CONTENT_RECT_CHANGED);
    CMD(APP_CMD_GAINED_FOCUS);
    CMD(APP_CMD_LOST_FOCUS);
    CMD(APP_CMD_CONFIG_CHANGED);
    CMD(APP_CMD_LOW_MEMORY);
    CMD(APP_CMD_START);
    CMD(APP_CMD_RESUME);
    CMD(APP_CMD_SAVE_STATE);
    CMD(APP_CMD_PAUSE);
    CMD(APP_CMD_STOP);
    CMD(APP_CMD_DESTROY);
  }
#undef CMD
}

extern "C" {

static void handle_cmd(android_app* androidApp, int32_t cmd) {
  VulkanApp* app = (VulkanApp*)androidApp->userData;

  LLOGD("handle_cmd(%s)", cmdToString(cmd));

  switch (cmd) {
  case APP_CMD_INIT_WINDOW:
    if (androidApp->window) {
      app->width_ = ANativeWindow_getWidth(androidApp->window);
      app->height_ = ANativeWindow_getHeight(androidApp->window);
      if (!app->ctx_)
        app->ctx_ = lvk::createVulkanContextWithSwapchain(androidApp->window, app->width_, app->height_, {});
      app->canRender_ = true;
    }
    return;
  case APP_CMD_GAINED_FOCUS:
    app->canRender_ = app->ctx_ != nullptr;
    return;
  case APP_CMD_LOST_FOCUS:
  case APP_CMD_TERM_WINDOW:
    app->canRender_ = false;
    return;
  case APP_CMD_DESTROY:
    return;
  }
}

static void resize_callback(ANativeActivity* activity, ANativeWindow* window) {
  LLOGD("resize_callback()");

  VulkanApp* app = (VulkanApp*)activity->instance;
  const int w = ANativeWindow_getWidth(window);
  const int h = ANativeWindow_getHeight(window);
  if (app->width_ != w || app->height_ != h) {
    app->width_ = w;
    app->height_ = h;
    if (app->ctx_) {
      app->ctx_->recreateSwapchain(w, h);
      LLOGD("Swapchain recreated");
    }
  }

  LLOGD("resize_callback()<-");
}

} // extern "C"
#endif // ANDROID

#if defined(ANDROID)
VulkanApp::VulkanApp(android_app* androidApp, const VulkanAppConfig& cfg) : androidApp_(androidApp), cfg_(cfg) {
#else
VulkanApp::VulkanApp(const VulkanAppConfig& cfg) : cfg_(cfg) {
#endif // ANDROID
  minilog::initialize(nullptr, {.threadNames = false});

  // find the content folder
  {
    using namespace std::filesystem;
#if defined(ANDROID)
    if (const char* externalStorage = std::getenv("EXTERNAL_STORAGE")) {
      folderThirdParty_ = (path(externalStorage) / "LVK" / "deps" / "src").string() + "/";
      folderContentRoot_ = (path(externalStorage) / "LVK" / "content").string() + "/";
    }
#else
    path subdir("third-party/content/");
    path dir = current_path();
    // find the content somewhere above our current build directory
    while (dir != current_path().root_path() && !exists(dir / subdir)) {
      dir = dir.parent_path();
    }
    if (!exists(dir / subdir)) {
      LLOGW("Cannot find the content directory. Run `deploy_content.py` before running this app.");
      LVK_ASSERT(false);
      return;
    }
    folderThirdParty_ = (dir / path("third-party/deps/src/")).string();
    folderContentRoot_ = (dir / subdir).string();
#endif // ANDROID
  }

#if defined(ANDROID)
  androidApp_->userData = this;
  androidApp_->onAppCmd = handle_cmd;

  fpsCounter_.printFPS_ = false;

  int events = 0;
  android_poll_source* source = nullptr;

  while (!androidApp_->destroyRequested && !ctx_) {
    // poll until a Window is created
    if (ALooper_pollOnce(0, nullptr, &events, (void**)&source) >= 0) {
      if (source)
        source->process(androidApp_, source);
    }
    LLOGD("Waiting for an Android window...");
  }

  if (!ctx_)
    return;

  androidApp_->activity->instance = this;
  androidApp_->activity->callbacks->onNativeWindowResized = resize_callback;
#else
  width_ = cfg.width;
  height_ = cfg.height;

  window_ = lvk::initWindow("Simple example", width_, height_, cfg.resizable);

  ctx_ = lvk::createVulkanContextWithSwapchain(window_,
                                               width_,
                                               height_,
                                               {
                                                   .enableValidation = true,
                                               });
#endif // ANDROID

  depthTexture_ = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_Z_F32,
      .dimensions = {(uint32_t)width_, (uint32_t)height_},
      .usage = lvk::TextureUsageBits_Attachment,
      .debugName = "Depth buffer",
  });

  imgui_ = std::make_unique<lvk::ImGuiRenderer>(
      *ctx_, (folderThirdParty_ + "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf").c_str(), 30.0f);
#if !defined(ANDROID)
  glfwSetWindowUserPointer(window_, this);

  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* window, int width, int height) {
    VulkanApp* app = (VulkanApp*)glfwGetWindowUserPointer(window);
    app->width_ = width;
    app->height_ = height;
    app->ctx_->recreateSwapchain(width, height);
  });
  glfwSetMouseButtonCallback(window_, [](GLFWwindow* window, int button, int action, int mods) {
    VulkanApp* app = (VulkanApp*)glfwGetWindowUserPointer(window);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      app->mouseState_.pressedLeft = action == GLFW_PRESS;
    }
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    const ImGuiMouseButton_ imguiButton = (button == GLFW_MOUSE_BUTTON_LEFT)
                                              ? ImGuiMouseButton_Left
                                              : (button == GLFW_MOUSE_BUTTON_RIGHT ? ImGuiMouseButton_Right : ImGuiMouseButton_Middle);
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)xpos, (float)ypos);
    io.MouseDown[imguiButton] = action == GLFW_PRESS;
    for (auto& cb : app->callbacksMouseButton) {
      cb(window, button, action, mods);
    }
  });
  glfwSetScrollCallback(window_, [](GLFWwindow* window, double dx, double dy) {
    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH = (float)dx;
    io.MouseWheel = (float)dy;
  });
  glfwSetCursorPosCallback(window_, [](GLFWwindow* window, double x, double y) {
    VulkanApp* app = (VulkanApp*)glfwGetWindowUserPointer(window);
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    ImGui::GetIO().MousePos = ImVec2(x, y);
    app->mouseState_.pos.x = static_cast<float>(x / width);
    app->mouseState_.pos.y = 1.0f - static_cast<float>(y / height);
  });
  glfwSetKeyCallback(window_, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
    VulkanApp* app = (VulkanApp*)glfwGetWindowUserPointer(window);
    const bool pressed = action != GLFW_RELEASE;
    if (key == GLFW_KEY_ESCAPE && pressed)
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_W)
      app->positioner_.movement_.forward_ = pressed;
    if (key == GLFW_KEY_S)
      app->positioner_.movement_.backward_ = pressed;
    if (key == GLFW_KEY_A)
      app->positioner_.movement_.left_ = pressed;
    if (key == GLFW_KEY_D)
      app->positioner_.movement_.right_ = pressed;
    if (key == GLFW_KEY_1)
      app->positioner_.movement_.up_ = pressed;
    if (key == GLFW_KEY_2)
      app->positioner_.movement_.down_ = pressed;

    app->positioner_.movement_.fastSpeed_ = (mods & GLFW_MOD_SHIFT) != 0;

    if (key == GLFW_KEY_SPACE) {
      app->positioner_.lookAt(app->cfg_.initialCameraPos, app->cfg_.initialCameraTarget, vec3(0.0f, 1.0f, 0.0f));
    }
    for (auto& cb : app->callbacksKey) {
      cb(window, key, scancode, action, mods);
    }
  });
#endif // !ANDROID
}

VulkanApp::~VulkanApp() {
  imgui_ = nullptr;
  depthTexture_ = nullptr;
  ctx_ = nullptr;
#if !defined(ANDROID)
  glfwDestroyWindow(window_);
  glfwTerminate();
#endif // !ANDROID
}

lvk::Format VulkanApp::getDepthFormat() const {
  return ctx_->getFormat(depthTexture_);
}

void VulkanApp::run(DrawFrameFunc drawFrame) {
  double timeStamp = glfwGetTime();
  float deltaSeconds = 0.0f;

#if defined(ANDROID)
  while (!androidApp_->destroyRequested) {
#else
  while (!glfwWindowShouldClose(window_)) {
#endif // !ANDROID
    if (fpsCounter_.tick(deltaSeconds)) {
      LLOGL("FPS: %.1f\n", fpsCounter_.getFPS());
    }
    const double newTimeStamp = glfwGetTime();
    deltaSeconds = static_cast<float>(newTimeStamp - timeStamp);
    timeStamp = newTimeStamp;

#if defined(ANDROID)
    android_poll_source* source = nullptr;
    const int result = ALooper_pollOnce(canRender_ ? 0 : -1, nullptr, nullptr, (void**)&source);

    if (result == ALOOPER_POLL_ERROR) {
      LLOGW("ALooper_pollOnce() returned an error");
      break;
    }

    if (source) {
      source->process(androidApp_, source);
    }
#else

#if defined(__APPLE__)
    // a hacky workaround for retina displays
    glfwGetWindowSize(window_, &width_, &height_);
#else
    glfwGetFramebufferSize(window_, &width_, &height_);
#endif // __APPLE__

    glfwPollEvents();
#endif // ANDROID

    if (!width_ || !height_)
      continue;
    const float ratio = width_ / (float)height_;

    positioner_.update(deltaSeconds, mouseState_.pos, ImGui::GetIO().WantCaptureMouse ? false : mouseState_.pressedLeft);

    if (ctx_ && canRender_) {
      drawFrame((uint32_t)width_, (uint32_t)height_, ratio, deltaSeconds);
    }
  }

  LLOGD("Terminating Android app...");
}

void VulkanApp::drawFPS() {
  if (const ImGuiViewport* v = ImGui::GetMainViewport()) {
    ImGui::SetNextWindowPos({v->WorkPos.x + v->WorkSize.x - 15.0f, v->WorkPos.y + 15.0f}, ImGuiCond_Always, {1.0f, 0.0f});
  }
  ImGui::SetNextWindowBgAlpha(0.30f);
  ImGui::SetNextWindowSize(ImVec2(ImGui::CalcTextSize("FPS : _______").x, 0));
  if (ImGui::Begin("##FPS",
                   nullptr,
                   ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove)) {
    ImGui::Text("FPS : %i", (int)fpsCounter_.getFPS());
    ImGui::Text("Ms  : %.1f", fpsCounter_.getFPS() > 0 ? 1000.0 / fpsCounter_.getFPS() : 0);
  }
  ImGui::End();
}
