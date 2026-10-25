
cmake_minimum_required(VERSION 3.25)
project(GG_Fall C)

add_executable(GG-Fall
    main.c
)

set_target_properties(GG-Fall PROPERTIES
    C_STANDARD 17
    C_EXTENSIONS ON
)

target_compile_options(GG-Fall PRIVATE
    -Wall
    -Wextra
    -Werror
)

xsdk_add_shaders(GG-Fall
    cube.vert
    cube.frag
)

xsdk_add_xex(GG-Fall)
