#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "VBSGGUF.h"
@class VBSNode;

NS_ASSUME_NONNULL_BEGIN
@interface VBSMetalCompiler : NSObject
@property(nonatomic, readonly) id<MTLDevice> device;
@property(nonatomic, readonly) id<MTLCommandQueue> queue;
- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (nullable id<MTLComputePipelineState>)linearPipelineForType:(VBSGGMLType)type error:(NSError **)error;
- (nullable id<MTLComputePipelineState>)primitivePipelineForNode:(VBSNode *)node error:(NSError **)error;
@end
NS_ASSUME_NONNULL_END

