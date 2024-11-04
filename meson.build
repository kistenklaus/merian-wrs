project(
    'merian-example',
    'cpp',
    version : '1.0.0',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
        'buildtype=release',
    ]
)

# Debug configuration
if get_option('buildtype').startswith('debug')
  add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
endif

# Dependencies
merian = dependency('merian', version: ['>=1.0.0'], fallback: ['merian', 'merian_dep'], default_options: {'shaderc': 'enabled'})
shader_generator = merian_subp.get_variable('shader_generator')

src_files = files(
    'src/main.cpp',
)
inc_dirs = [
]
shader_generator.process('src/compute_sum.comp')

exe = executable(
    'merian-example',
    src_files,
    dependencies: [
        merian,
    ],
    include_directories: inc_dirs,
    install : true
)
