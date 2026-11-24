/* alchemist_list.c — print the Alchemist vector-add kernel listing.
 *
 * Usage: alchemist_list [output-file]
 * With no argument the listing goes to stdout. Used by
 * Makefile.alchemist to regenerate vbs_alchemist_vecadd.list.
 */

#include <stdio.h>

#include "vbs_alchemist_vecadd.h"

int main(int argc, char **argv)
{
    FILE *out = stdout;

    if (argc > 2) {
        fprintf(stderr, "usage: %s [output-file]\n", argv[0]);
        return 1;
    }
    if (argc == 2) {
        out = fopen(argv[1], "w");
        if (out == NULL) {
            fprintf(stderr, "cannot open %s for writing\n", argv[1]);
            return 1;
        }
    }
    vbs_alchemist_vecadd_list(out);
    if (out != stdout)
        fclose(out);
    return 0;
}
