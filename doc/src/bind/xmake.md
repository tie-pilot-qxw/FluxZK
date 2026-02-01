# xmake Build

The overall xmake build is the same as before, but note that we now build a library.

## Shared Library

Just add `set_kind("shared")`.

## Static Library

First add `set_kind("static")`. By default, xmake does not automatically link some nvcc-generated code into a static library. Since the later Rust build does not invoke nvcc, this can cause link errors. Enable devlink manually with `add_values("cuda.build.devlink", true)`.
