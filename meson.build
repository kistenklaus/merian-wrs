project(
    'merian-example',
    ['cpp', 'c'],
    version : '1.0.0',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
        'buildtype=release',
    ]
)


# renderdoc = declare_dependency(
#   link_args: ['-L/usr/bin/librenderdoc.so', '-lrenderdoc'],  # Linker flags
# )

# add_global_arguments('-fconcepts-diagnostics-depth=5', language : 'cpp')

# Debug configuration
if get_option('buildtype').startswith('debug')
  add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
endif

# Dependencies
merian_subp = subproject('merian')
merian = merian_subp.get_variable('merian_dep')
shader_generator = merian_subp.get_variable('shader_generator')

src_files = []
inc_dirs = []

subdir('src')


exe = executable(
    'merian-example',
    src_files,
    dependencies: [
        # renderdoc,
        merian,
    ],
    include_directories: inc_dirs,
    install : true
)
