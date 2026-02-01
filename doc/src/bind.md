# Rust Bind

## Overview

Overall, the Rust binding process has four steps:

1. Wrap C++ classes with a C-style API
2. Use xmake to build the C API and C++ classes into a static/shared library
3. Use `build.rs` to run xmake and `bindgen` to generate bindings automatically
4. Use `include!(concat!(env!("OUT_DIR"), "/bindings.rs"));` in Rust to import the C API and call it with `unsafe`

## Current Issues

The current approach converts C++ to C, so template support requires manual adjustments. One option is to inject macros at compile time via xmake to pass template parameters from Rust to C++. Another option is to try bindgen's C++ support, but official documentation is limited.
