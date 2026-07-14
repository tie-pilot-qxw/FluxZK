add_requires("doctest")
add_rules("mode.debug", "mode.release")

option("msm_window_size")
    set_default("22")
    set_showmenu(true)
    set_description("BN254 MSM window size for artifact benchmark")
option_end()

option("msm_precompute")
    set_default("2")
    set_showmenu(true)
    set_description("BN254 MSM precompute interval for artifact benchmark")
option_end()

option("msm_batch_size")
    set_default("4")
    set_showmenu(true)
    set_description("BN254 MSM batch size for artifact benchmark")
option_end()

option("msm_batch_per_run")
    set_default("2")
    set_showmenu(true)
    set_description("BN254 MSM batch-per-run for artifact benchmark")
option_end()

option("msm_parts")
    set_default("8")
    set_showmenu(true)
    set_description("BN254 MSM partition count for artifact benchmark")
option_end()

option("msm_config_file")
    set_default("")
    set_showmenu(true)
    set_description("Generated BN254 MSM explicit-instantiation file")
option_end()

option("msm_warmups")
    set_default("0")
    set_showmenu(true)
    set_description("Warmup runs before timing the BN254 MSM artifact benchmark")
option_end()

-- Custom rule to generate asm and populate template
rule("mont-gen-asm")
    set_extensions(".template")
    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        batchcmds:show_progress(opt.progress, '${color.build.object}templating from %s', sourcefile)
        batchcmds:execv("python3 mont/src/gen_asm.py", {sourcefile, target:targetfile()})
    end)
    on_link(function (target) end)

target("mont.cuh")
    add_files("mont/src/*.template")
    add_rules("mont-gen-asm")
    set_targetdir("mont/src")

target("test-mont")
    set_languages(("c++17"))
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_files("mont/tests/main.cu")
    add_packages("doctest")

target("bench-mont")
    set_languages(("c++17"))
    add_cugencodes("native")
    add_options("-lineinfo")
    add_options("--expt-relaxed-constexpr")
    add_files("mont/tests/bench.cu")

target("bench-mont0")
    add_deps("mont.cuh")
    add_options("-lineinfo")
    add_files("mont/tests/bench0.cu")

target("cuda_msm")
    set_kind("static")
    add_values("cuda.build.devlink", true)

    set_languages(("c++20"))
    add_files("msm/src/fast_compile/*.cu")
    add_files("wrapper/msm/c_api/msm_c_api.cu")
    add_headerfiles("wrapper/msm/c_api/*.h")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")

    set_targetdir("lib")

target("transpose")
    set_languages("c++17")
    set_optimize("fastest")
    add_files("ntt/src/transpose/*.cpp")
    add_cxflags("-mavx2")
    add_cxflags("-march=native")
    add_links("pthread")


includes("ntt")
includes("wrapper")
includes("msm")
