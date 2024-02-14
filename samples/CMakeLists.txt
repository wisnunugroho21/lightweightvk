# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME "LVK Samples")

if(MSVC)
  add_definitions(-D_CONSOLE)
else()
  add_compile_options(-Wno-deprecated-volatile)
endif()

if(WIN32)
  add_definitions("-DVK_USE_PLATFORM_WIN32_KHR=1")
  add_definitions("-DNOMINMAX")
endif()

macro(ADD_DEMO app)
  if(NOT ANDROID)
    add_executable(${app} "${app}.cpp")
    lvk_set_cxxstd(${app} 20)
    lvk_set_folder(${app} ${PROJECT_NAME})
    target_link_libraries(${app} PRIVATE LVKLibrary)
    target_link_libraries(${app} PRIVATE bc7enc)
    target_link_libraries(${app} PRIVATE meshoptimizer)
    target_link_libraries(${app} PRIVATE tinyobjloader)
    target_link_libraries(${app} PRIVATE glfw)
    if(UNIX AND NOT APPLE)
      target_link_libraries(${app} PRIVATE EGL)
    endif()
    target_link_libraries(${app} PRIVATE LVKstb)
  endif()

  # Android
  if(LVK_WITH_SAMPLES_ANDROID)
    if(NOT DEFINED ENV{ANDROID_NDK})
      message(FATAL_ERROR "ANDROID_NDK environment variable is not set")
    endif()

    # One CMakeLists file is used for generation Android Project and for building native library.
    # The native library is built only if ANDROID is defined, otherwise Android project is generated.
    if(NOT ANDROID)
      file(COPY "${LVK_ROOT_DIR}/android/build.gradle"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}")
      file(COPY "${LVK_ROOT_DIR}/android/gradle.properties"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}")
      file(COPY "${LVK_ROOT_DIR}/android/gradlew"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}")
      file(COPY "${LVK_ROOT_DIR}/android/gradlew.bat"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}")
      file(COPY "${LVK_ROOT_DIR}/android/gradle/wrapper/gradle-wrapper.jar"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/gradle/wrapper")
      file(COPY "${LVK_ROOT_DIR}/android/gradle/wrapper/gradle-wrapper.properties"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/gradle/wrapper")

      set(APP_NAME ${app})
      configure_file("${LVK_ROOT_DIR}/android/settings.gradle.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/settings.gradle")
      configure_file("${LVK_ROOT_DIR}/android/app/build.gradle.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/build.gradle")
      configure_file("${LVK_ROOT_DIR}/android/app/AndroidManifest.xml.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/AndroidManifest.xml")
      configure_file("${LVK_ROOT_DIR}/android/app/MainActivity.java.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/java/org/lvk/samples/MainActivity.java")

      file(COPY "${LVK_ROOT_DIR}/third-party/deps/src/android-validation-layers/arm64-v8a/libVkLayer_khronos_validation.so"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/jniLibs/arm64-v8a")
      file(COPY "${LVK_ROOT_DIR}/third-party/deps/src/android-validation-layers/armeabi-v7a/libVkLayer_khronos_validation.so"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/jniLibs/armeabi-v7a")
      file(COPY "${LVK_ROOT_DIR}/third-party/deps/src/android-validation-layers/x86/libVkLayer_khronos_validation.so"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/jniLibs/x86")
      file(COPY "${LVK_ROOT_DIR}/third-party/deps/src/android-validation-layers/x86_64/libVkLayer_khronos_validation.so"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/jniLibs/x86_64")         
    endif()

    if(ANDROID)
      cmake_path(SET NDK_PATH $ENV{ANDROID_NDK})
      add_library(lvk_android_native_${app} SHARED 
                  "${app}.cpp"
                  "${NDK_PATH}/sources/android/native_app_glue/android_native_app_glue.c")
      lvk_set_cxxstd(lvk_android_native_${app} 20)
      target_include_directories(lvk_android_native_${app} PRIVATE
                                 "${NDK_PATH}/sources/android/native_app_glue")
      target_link_libraries(lvk_android_native_${app} PRIVATE 
                            LVKLibrary LVKstb bc7enc meshoptimizer tinyobjloader android log)
    endif()
  endif()
endmacro()

macro(ADD_DEMO_SOURCES app srcs)
  if(ANDROID)
    target_sources(lvk_android_native_${app} PRIVATE ${srcs})
  else()
    target_sources(${app} PRIVATE ${srcs})
  endif()
endmacro()

macro(ADD_DEMO_LINK_LIBRARIES app libs)
  if(ANDROID)
    target_link_libraries(lvk_android_native_${app} PRIVATE ${libs})
  else()
    target_link_libraries(${app} PRIVATE ${libs})
  endif()
endmacro()

ADD_DEMO("HelloTriangle")

ADD_DEMO("Tiny")

ADD_DEMO("Tiny_Mesh")

ADD_DEMO("Tiny_MeshLarge")
ADD_DEMO_SOURCES("Tiny_MeshLarge"
  "${LVK_ROOT_DIR}/third-party/deps/src/3D-Graphics-Rendering-Cookbook/shared/UtilsCubemap.cpp")
ADD_DEMO_SOURCES("Tiny_MeshLarge"
  "${LVK_ROOT_DIR}/third-party/deps/src/imgui/imgui_demo.cpp")
ADD_DEMO_LINK_LIBRARIES("Tiny_MeshLarge" ktx)
