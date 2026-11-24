/* vbs_cc.c — system-compiler invocation (fork/exec, no shell).
 *
 * Compiles one C source file into a shared object by spawning the
 * system compiler directly (fork/execvp, argv array — no shell, so
 * no quoting or injection surface). The compiler's stderr is piped
 * back to the caller; stdout goes to /dev/null. The parent drains
 * the pipe fully so a chatty compiler can never block on a full
 * pipe, then waits for the child and reports its exit status. */

#include "vbs_cc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

const char *vbs_cc_default(void) {
    const char *cc = getenv("CC");
    return (cc && *cc) ? cc : "cc";
}

#if defined(_WIN32)
/* Windows has no fork/exec: compiling a candidate .so needs a Windows
 * supervisor (CreateProcess) that does not exist yet. The loop and the
 * fuser fail gracefully with this message instead of crashing. */
int vbs_cc_compile_so(const char *cc, const char *src, const char *so,
                      const char *const *extra_argv, size_t n_extra,
                      char *err_out, size_t err_cap) {
    (void)cc;
    (void)src;
    (void)so;
    (void)extra_argv;
    (void)n_extra;
    if (err_out && err_cap)
        snprintf(err_out, err_cap,
                 "system-compiler invocation is POSIX-only in this build");
    return -1;
}
#else

/* Drain a pipe into nowhere (used after the error buffer is full, so
 * the child never blocks on a full pipe). */
static void drain_pipe(int fd) {
    char scratch[4096];
    while (read(fd, scratch, sizeof(scratch)) > 0) { }
}

int vbs_cc_compile_so(const char *cc, const char *src, const char *so,
                      const char *const *extra_argv, size_t n_extra,
                      char *err_out, size_t err_cap) {
    if (err_out && err_cap) err_out[0] = '\0';
    if (!cc) cc = vbs_cc_default();

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        if (err_out && err_cap) snprintf(err_out, err_cap, "pipe() failed");
        return -1;
    }
    pid_t pid = fork();
    if (pid < 0) {
        if (err_out && err_cap) snprintf(err_out, err_cap, "fork() failed");
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        /* Child: stderr -> pipe, stdout -> /dev/null. */
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        FILE *devnull = fopen("/dev/null", "w");
        if (devnull) {
            dup2(fileno(devnull), STDOUT_FILENO);
            fclose(devnull);
        }
        const char **argv = (const char **)calloc(8 + n_extra, sizeof(char *));
        if (!argv) _exit(127);
        size_t k = 0;
        argv[k++] = cc;
        argv[k++] = "-O3";
        argv[k++] = "-shared";
        argv[k++] = "-fPIC";
        argv[k++] = "-o";
        argv[k++] = so;
        argv[k++] = src;
        for (size_t i = 0; i < n_extra; i++) argv[k++] = extra_argv[i];
        argv[k] = NULL;
        /* execvp needs char *const *; our strings are never modified. */
        execvp(cc, (char *const *)argv);
        _exit(127);
    }
    close(pipefd[1]);
    size_t n = 0;
    if (err_out && err_cap) {
        ssize_t r;
        while (n + 1 < err_cap &&
               (r = read(pipefd[0], err_out + n, err_cap - 1 - n)) > 0) {
            n += (size_t)r;
        }
        drain_pipe(pipefd[0]);
        err_out[n] = '\0';
    } else {
        drain_pipe(pipefd[0]);
    }
    close(pipefd[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) { }
    if (WIFEXITED(status)) {
        int rc = WEXITSTATUS(status);
        if (rc == 127 && err_out && err_cap && n == 0) {
            snprintf(err_out, err_cap, "failed to exec compiler '%s'", cc);
        }
        return rc;
    }
    if (err_out && err_cap)
        snprintf(err_out, err_cap, "compiler killed by signal %d", WTERMSIG(status));
    return -1;
}
#endif /* _WIN32 */
