# Verum-Bespoke-Singularity
Pure-C GPU LLM inference engine.
Verum builds one binary, verum-slice, that reads GGUF models and tokenizer.json, with commands --inspect, --compile, --benchmark, and --generate.
## Backends
Single backend abstraction (vbs_backend_ops_t) covering Metal, CUDA, HIP, and CPU. The same C source is rebuilt per OS, with the GPU backend selected at runtime where a driver is present. CUDA and HIP are loaded at runtime, not link-time locked.
## Backend status
Metal and CUDA are the fuller paths. HIP embedding is currently a stub and is work in progress.
## Codebase
About 4,630 lines across src/*.c and include/*.h, C11.
