set(TARGET worker)

include(ExternalProject)

# Set the URL and tag of the Mongoose repository on Github
set(MONGOOSE_GIT_URL "https://github.com/cesanta/mongoose.git")
set(MONGOOSE_GIT_TAG "master")

# Define the ExternalProject to download and build Mongoose
ExternalProject_Add(mongoose_project
    GIT_REPOSITORY ${MONGOOSE_GIT_URL}
    GIT_TAG ${MONGOOSE_GIT_TAG}
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/mongoose"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    BUILD_IN_SOURCE ON
    INSTALL_COMMAND ""
)

add_executable(${TARGET} 
  worker.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/mongoose/src/mongoose_project/mongoose.c"
)
target_include_directories(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/mongoose/src/mongoose_project")

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11 )
