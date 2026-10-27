#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <strings.h>
#include "vbs_learn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
static void clean(char *s){ for(;*s;s++){ if(*s=='\t'||*s=='\n'||*s=='\r') *s=' '; } }
static int word_in(const char *hay, const char *word, size_t n){
    for(const char *p=hay; *p; ){
        while(*p && !isalnum((unsigned char)*p)) p++;
        const char *s=p; while(*p && isalnum((unsigned char)*p)) p++;
        if((size_t)(p-s)==n && strncasecmp(s,word,n)==0) return 1;
    }
    return 0;
}
static double overlap(const char *a, const char *b){
    int match=0, total=0;
    for(const char *p=a; *p; ){
        while(*p && !isalnum((unsigned char)*p)) p++;
        const char *s=p; while(*p && isalnum((unsigned char)*p)) p++;
        if(p==s) break;
        total++;
        if(word_in(b,s,(size_t)(p-s))) match++;
    }
    return total ? (double)match/total : 0.0;
}
bool vbs_learn_append(const char *path, const char *prompt, const char *correct){
    FILE *f=fopen(path,"a"); if(!f) return false;
    char *p=strdup(prompt), *c=strdup(correct);
    if(!p||!c){ if(p) free(p); if(c) free(c); fclose(f); return false; }
    clean(p); clean(c); fprintf(f,"%s\t%s\n",p,c); free(p); free(c); fclose(f); return true;
}
double vbs_learn_recall(const char *path, const char *prompt, char *out_correct, size_t out_cap){
    FILE *f=fopen(path,"r"); if(!f) return 0.0;
    char line[8192]; double best=0.0; out_correct[0]='\0';
    while(fgets(line,sizeof line,f)){
        char *tab=strchr(line,'\t'); if(!tab) continue; *tab='\0';
        char *corr=tab+1; corr[strcspn(corr,"\r\n")]='\0';
        double s=overlap(prompt,line);
        if(s>best){ best=s; snprintf(out_correct,out_cap,"%s",corr); }
    }
    fclose(f); return best;
}
