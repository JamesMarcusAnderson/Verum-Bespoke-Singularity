#include "vbs_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
        "Verum Bespoke Singularity — Pure C Multi-Backend LLM Engine\n"
        "\nusage: verum-slice [--inspect|--compile] [--window-mib N] model.gguf\n"
        "\nenv: VBS_BACKEND=cuda|hip|metal|cpu\n");
}

int main(int argc, const char *argv[]) {
    if (argc < 3) { usage(); return 64; }
    const char *mode = argv[1];
    size_t window_mib = 256;
    int ai = 2;

    if (ai + 1 < argc && strcmp(argv[ai], "--window-mib") == 0) {
        window_mib = (size_t)strtoull(argv[ai + 1], NULL, 10);
        ai += 2;
    }
    if (ai >= argc) { usage(); return 64; }

    char err[512] = {0};
    vbs_engine_t *engine = vbs_engine_create(argv[ai], window_mib, err, sizeof(err));
    if (!engine) { fprintf(stderr, "load failed: %s\n", err); return 1; }

    vbs_engine_print_inspection(engine);
    if (strcmp(mode, "--inspect") == 0) { vbs_engine_destroy(engine); return 0; }
    if (strcmp(mode, "--compile") == 0) {
        if (!vbs_engine_compile(engine, err, sizeof(err))) {
            fprintf(stderr, "compile refused: %s\n", err);
            vbs_engine_destroy(engine);
            return 2;
        }
        vbs_engine_destroy(engine);
        return 0;
    }
    usage();
    vbs_engine_destroy(engine);
    return 64;
}
