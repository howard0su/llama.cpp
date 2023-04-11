set(TARGET worker)

add_executable(${TARGET} 
  worker.cpp
  engine.cpp
  cJSON.c
  mongoose.c
)

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11 )
