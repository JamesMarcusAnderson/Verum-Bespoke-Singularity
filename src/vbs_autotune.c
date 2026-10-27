#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "vbs_autotune.h"
#include "vbs_backend.h"
#include "vbs_block.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif
static double now_sec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec/1e9; }
static int gpu_available(const char *label, const vbs_backend_ops_t *ops){
    if(!ops||!ops->init) return 0;
    return ops->init() ? 1 : 0;
}
int vbs_autotune_detect(vbs_autotune_t *cfg){
    memset(cfg,0,sizeof *cfg);
#if defined(_WIN32)
    SYSTEM_INFO si; GetSystemInfo(&si); cfg->cores=(int)si.dwNumberOfProcessors;
    MEMORYSTATUSEX ms; ms.dwLength=sizeof ms; GlobalMemoryStatusEx(&ms); cfg->ram_bytes=ms.ullTotalPhys;
#elif defined(__APPLE__)
    size_t len=sizeof(int); int cores=0; sysctlbyname("hw.ncpu",&cores,&len,NULL,0); cfg->cores=cores;
    uint64_t mem=0; len=sizeof(mem); sysctlbyname("hw.memsize",&mem,&len,NULL,0); cfg->ram_bytes=mem;
#else
    long c=sysconf(_SC_NPROCESSORS_ONLN); cfg->cores=c>0?(int)c:1;
    FILE *f=fopen("/proc/meminfo","r");
    if(f){ char line[256]; while(fgets(line,sizeof line,f)){ unsigned long kb=0; if(sscanf(line,"MemTotal: %lu kB",&kb)==1){ cfg->ram_bytes=kb*1024UL; break; } } fclose(f); }
#endif
    if(cfg->cores<=0) cfg->cores=1;
    if(cfg->ram_bytes==0) cfg->ram_bytes=1ULL<<30;
    cfg->threads=cfg->cores;
    /* backend: first available GPU, else CPU */
    if(gpu_available("CUDA",&g_vbs_cuda_backend)) strcpy(cfg->backend,"CUDA");
    else if(gpu_available("HIP",&g_vbs_hip_backend)) strcpy(cfg->backend,"HIP");
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    else if(gpu_available("Metal",&g_vbs_metal_backend)) strcpy(cfg->backend,"Metal");
#endif
    else strcpy(cfg->backend,"CPU");
    size_t win = cfg->ram_bytes/8/1024/1024;
    if(win<64) win=64;
    if(win>1024) win=1024;
    cfg->window_mib=win;
    snprintf(cfg->fingerprint,sizeof cfg->fingerprint,"%s-%d-%llu",cfg->backend,cfg->cores,(unsigned long long)(cfg->ram_bytes>>20));
    return 1;
}
double vbs_autotune_benchmark(void){
    vbs_block_t *blk=vbs_block_create(8,16,3,32,12345u);
    int toks[3]={1,2,3};
    double t0=now_sec();
    for(int i=0;i<100;i++) vbs_block_train_step(blk,toks,5,0.05f);
    double dt=now_sec()-t0;
    vbs_block_free(blk);
    return dt>0 ? 100.0/dt : 0;
}
int vbs_autotune_load(const char *path, vbs_autotune_t *out){
    FILE *f=fopen(path,"r"); if(!f) return 0;
    char backend[32], fp[128]; int threads,cores; size_t win; unsigned long long ram; double score;
    int n=fscanf(f,"backend=%31s threads=%d window_mib=%zu cores=%d ram_mb=%llu score=%lf fingerprint=%127s",backend,&threads,&win,&cores,&ram,&score,fp);
    fclose(f); if(n!=7) return 0;
    strcpy(out->backend,backend); out->threads=threads; out->window_mib=win; out->cores=cores; out->ram_bytes=ram*1024*1024; out->score_epochs_per_sec=score; strcpy(out->fingerprint,fp);
    return 1;
}
int vbs_autotune_run(const char *config_path, vbs_autotune_t *out){
    vbs_autotune_t cur; vbs_autotune_detect(&cur);
    vbs_autotune_t old;
    if(vbs_autotune_load(config_path,&old) && strcmp(old.fingerprint,cur.fingerprint)==0){
        printf("autotune: learned config reused for %s (backend=%s threads=%d window=%zu MiB score=%.1f epochs/s)\n",cur.fingerprint,old.backend,old.threads,old.window_mib,old.score_epochs_per_sec);
        *out=old; return 2; /* reused */
    }
    cur.score_epochs_per_sec=vbs_autotune_benchmark();
    FILE *f=fopen(config_path,"w");
    if(f){ fprintf(f,"backend=%s threads=%d window_mib=%zu cores=%d ram_mb=%llu score=%.2f fingerprint=%s\n",cur.backend,cur.threads,cur.window_mib,cur.cores,(unsigned long long)(cur.ram_bytes>>20),cur.score_epochs_per_sec,cur.fingerprint); fclose(f); }
    printf("autotune: backend=%s cores=%d threads=%d ram=%llu MiB window=%zu MiB score=%.1f epochs/s fingerprint=%s\n",cur.backend,cur.cores,cur.threads,(unsigned long long)(cur.ram_bytes>>20),cur.window_mib,cur.score_epochs_per_sec,cur.fingerprint);
    *out=cur; return 1; /* newly tuned */
}
