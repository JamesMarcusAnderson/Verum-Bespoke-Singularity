#import "VBSFabric.h"

static NSString*const FabricError=@"Verum.Fabric";
static NSError*ferr(NSString*s){return [NSError errorWithDomain:FabricError code:1 userInfo:@{NSLocalizedDescriptionKey:s}];}

@implementation VBSFabric { NSMapTable<id<MTLDevice>,id<MTLCommandQueue>>*_queues; }
- (instancetype)initWithDevices:(NSArray<id<MTLDevice>>*)d { if((self=[super init])){_devices=[d copy];_queues=[NSMapTable strongToStrongObjectsMapTable];for(id<MTLDevice>x in d)[_queues setObject:[x newCommandQueue] forKey:x];}return self; }
- (id<MTLBuffer>)viewOrCopyBuffer:(id<MTLBuffer>)src toDevice:(id<MTLDevice>)dst wait:(BOOL)wait error:(NSError**)error {
    if(src.device==dst)return src;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    id<MTLBuffer>remote=nil;
    if([src respondsToSelector:@selector(newRemoteBufferViewForDevice:)] &&
       [src.device respondsToSelector:@selector(peerGroupID)] &&
       src.device.peerGroupID!=0 && src.device.peerGroupID==dst.peerGroupID){
        remote=[src newRemoteBufferViewForDevice:dst];
    }
#pragma clang diagnostic pop
    if(remote)return remote;
    // Metal resources belong to the device that created them.  The fallback
    // therefore returns the destination resource explicitly; it never loses it
    // in a void method as the uploaded UnifiedHeap implementation did.
    id<MTLBuffer>host=[src.device newBufferWithLength:src.length options:MTLResourceStorageModeShared];
    if(!host){if(error)*error=ferr(@"Unable to allocate cross-device staging buffer");return nil;}
    id<MTLCommandQueue>sq=[_queues objectForKey:src.device];id<MTLCommandBuffer>read=[sq commandBuffer];id<MTLBlitCommandEncoder>rb=[read blitCommandEncoder];[rb copyFromBuffer:src sourceOffset:0 toBuffer:host destinationOffset:0 size:src.length];[rb endEncoding];[read commit];[read waitUntilCompleted];if(read.status==MTLCommandBufferStatusError){if(error)*error=read.error;return nil;}
    id<MTLBuffer>dstStage=[dst newBufferWithLength:src.length options:MTLResourceStorageModeShared];if(!dstStage){if(error)*error=ferr(@"Unable to allocate destination staging buffer");return nil;}memcpy(dstStage.contents,host.contents,src.length);
    id<MTLBuffer>out=[dst newBufferWithLength:src.length options:MTLResourceStorageModePrivate];if(!out){if(error)*error=ferr(@"Unable to allocate destination buffer");return nil;}id<MTLCommandQueue>dq=[_queues objectForKey:dst];id<MTLCommandBuffer>write=[dq commandBuffer];id<MTLBlitCommandEncoder>wb=[write blitCommandEncoder];[wb copyFromBuffer:dstStage sourceOffset:0 toBuffer:out destinationOffset:0 size:src.length];[wb endEncoding];[write commit];if(wait){[write waitUntilCompleted];if(write.status==MTLCommandBufferStatusError){if(error)*error=write.error;return nil;}}return out;
}
@end
