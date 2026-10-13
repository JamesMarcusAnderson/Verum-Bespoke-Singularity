#include "vbs_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr, "usage: verum-slice [--inspect|--compile] [--window-mib N] model.gguf\n");
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        usage();
        return 64;
    }

    const char *mode = argv[1];
    size_t window_mib = 256;
    int arg_idx = 2;

    if (arg_idx + 1 < argc && strcmp(argv[arg_idx], "--window-mib") == 0) {
        window_mib = (size_t)strtoull(argv[arg_idx + 1], NULL, 10);
        arg_idx += 2;
    }

    if (arg_idx >= argc) {
        usage();
        return 64;
    }

    const char *model_path = argv[arg_idx];
    char err_buf[512] = {0};

    vbs_engine_t *engine = vbs_engine_create(model_path, window_mib, err_buf, sizeof(err_buf));
    if (!engine) {
        fprintf(stderr, "load failed: %s\n", err_buf);
        return 1;
    }

    vbs_engine_print_inspection(engine);

    if (strcmp(mode, "--inspect") == 0) {
        vbs_engine_destroy(engine);
        return 0;
    }

    if (strcmp(mode, "--compile") == 0) {
        if (!vbs_engine_compile(engine, err_buf, sizeof(err_buf))) {
            fprintf(stderr, "compile refused: %s\n", err_buf);
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
