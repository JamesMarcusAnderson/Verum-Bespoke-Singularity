#import <Foundation/Foundation.h>
#import "Engine.h"

static void usage(void){fprintf(stderr,"usage: verum-slice [--inspect|--compile] [--window-mib N] model.gguf\n");}
int main(int argc,const char*argv[]){@autoreleasepool{if(argc<3){usage();return 64;}NSString*mode=@(argv[1]);NSUInteger window=256;int i=2;if(i+1<argc&&!strcmp(argv[i],"--window-mib")){window=(NSUInteger)strtoull(argv[i+1],NULL,10);i+=2;}if(i>=argc){usage();return 64;}NSError*e=nil;Engine*engine=[[Engine alloc]initWithModelPath:@(argv[i]) windowMiB:window error:&e];if(!engine){fprintf(stderr,"load failed: %s\n",e.localizedDescription.UTF8String);return 1;}[engine printInspection];if([mode isEqualToString:@"--inspect"])return 0;if([mode isEqualToString:@"--compile"]){if(![engine compileDiscoveredHome:&e]){fprintf(stderr,"compile refused: %s\n",e.localizedDescription.UTF8String);return 2;}return 0;}usage();return 64;}}

