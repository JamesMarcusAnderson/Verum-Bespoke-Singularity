#import "Engine.h"
#import "VBSGGUF.h"
#import "VBSGraph.h"
#import "VBSMetal.h"
#import "VBSSliceExecutor.h"
#import "VBSFabric.h"

@implementation Engine { VBSGGUFModel*_model;VBSGraph*_graph;NSArray<VBSMetalCompiler*>*_compilers;NSArray<VBSSliceExecutor*>*_executors;VBSFabric*_fabric;NSUInteger _windowBytes; }
- (instancetype)initWithModelPath:(NSString*)path windowMiB:(NSUInteger)mib error:(NSError**)error {
    if(!(self=[super init]))return nil;_model=[[VBSGGUFModel alloc]initWithPath:path error:error];if(!_model)return nil;
    _windowBytes=MAX((NSUInteger)1,mib)*1024*1024;NSArray*devices=MTLCopyAllDevices();if(!devices.count){id<MTLDevice>d=MTLCreateSystemDefaultDevice();if(d)devices=@[d];}if(!devices.count){if(error)*error=[NSError errorWithDomain:@"Verum.Engine" code:1 userInfo:@{NSLocalizedDescriptionKey:@"No Metal device"}];return nil;}
    NSMutableArray*c=[NSMutableArray array],*x=[NSMutableArray array];for(id<MTLDevice>d in devices){VBSMetalCompiler*mc=[[VBSMetalCompiler alloc]initWithDevice:d];[c addObject:mc];VBSSliceExecutor*xe=[[VBSSliceExecutor alloc]initWithModel:_model compiler:mc windowBytes:_windowBytes];if(!xe){if(error)*error=[NSError errorWithDomain:@"Verum.Engine" code:1 userInfo:@{NSLocalizedDescriptionKey:@"Unable to allocate slice windows"}];return nil;}[x addObject:xe];}_compilers=c;_executors=x;_fabric=[[VBSFabric alloc]initWithDevices:devices];return self;
}
- (void)printInspection { printf("GGUF v%u | %lu tensors | %lu metadata | window %lu MiB | %lu GPU(s)\n",_model.version,(unsigned long)_model.tensors.count,(unsigned long)_model.metadata.count,(unsigned long)(_windowBytes/1024/1024),(unsigned long)_compilers.count);NSString*a=[_model stringForMetadataKey:@"general.architecture"]?:@"(not declared)";printf("declared architecture: %s\n",a.UTF8String);for(VBSMetalCompiler*c in _compilers)printf("GPU: %s | working set %llu MiB | max buffer %llu MiB\n",c.device.name.UTF8String,c.device.recommendedMaxWorkingSetSize/1024/1024,c.device.maxBufferLength/1024/1024);NSMutableDictionary*types=[NSMutableDictionary dictionary];for(VBSTensor*t in _model.tensors)types[@(t.type)]=@([types[@(t.type)] unsignedIntegerValue]+1);printf("encodings:");for(NSNumber*k in [[types allKeys]sortedArrayUsingSelector:@selector(compare:)])printf(" %u=%lu",k.unsignedIntValue,(unsigned long)[types[k] unsignedIntegerValue]);printf("\n"); }
- (BOOL)compileDiscoveredHome:(NSError**)error { _graph=[[[VBSGraphCompiler alloc]init]compileModel:_model error:error];if(!_graph)return NO;for(VBSMetalCompiler*c in _compilers){for(VBSNode*n in _graph.nodes){id<MTLComputePipelineState>p=n.kind==VBSOpLinear?[c linearPipelineForType:n.weight.type error:error]:[c primitivePipelineForNode:n error:error];if(!p)return NO;}}printf("graph %s | %lu inferred operations | generated MSL compiled on %lu GPU(s)\n",_graph.fingerprint.UTF8String,(unsigned long)_graph.nodes.count,(unsigned long)_compilers.count);return YES; }
@end
