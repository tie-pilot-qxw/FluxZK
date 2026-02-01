# Memory Layout

Halo2 uses 64-bit integers per word, while CUDA uses 32-bit integers per word. Since both are little-endian and the target system is also little-endian, the in-memory layout can be considered identical. Therefore, field elements do not need conversion when moving from Halo2 to CUDA.

![image-20240915120825413](memory_layout.assets/image-20240915120825413.png)
