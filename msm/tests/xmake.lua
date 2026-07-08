target("test-bn254")
    if is_mode("debug") then
        set_symbols("debug")
    end
    set_languages(("c++20"))
    add_files("bn254.cu")
    add_cugencodes("native")
    add_packages("doctest")

target("test-msm")
    set_languages(("c++20"))
    add_files("msm.cu")
    add_files("../src/fast_compile/msm_bn254_22_2_f.cu")
    add_cugencodes("native")

target("bench-msm-bn254-ae")
    local msm_window_size = get_config("msm_window_size") or "22"
    local msm_precompute = get_config("msm_precompute") or "2"
    local msm_batch_size = get_config("msm_batch_size") or "4"
    local msm_batch_per_run = get_config("msm_batch_per_run") or "2"
    local msm_parts = get_config("msm_parts") or "8"
    local msm_config_file = get_config("msm_config_file") or ""
    set_languages(("c++20"))
    add_defines(
        "MSM_WINDOW_SIZE=" .. msm_window_size,
        "MSM_PRECOMPUTE=" .. msm_precompute,
        "MSM_BATCH_SIZE=" .. msm_batch_size,
        "MSM_BATCH_PER_RUN=" .. msm_batch_per_run,
        "MSM_PARTS=" .. msm_parts
    )
    add_files("msm.cu")
    if msm_config_file ~= "" then
        add_files(msm_config_file)
    else
        add_files("../src/fast_compile/msm_bn254_22_2_f.cu")
    end
    add_cugencodes("native")
