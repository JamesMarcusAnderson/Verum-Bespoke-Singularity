#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
@class VBSGGUFModel,VBSTensor,VBSMetalCompiler;
NS_ASSUME_NONNULL_BEGIN
@interface VBSSliceExecutor : NSObject
@property(nonatomic, readonly) NSUInteger windowBytes;
- (instancetype)initWithModel:(VBSGGUFModel*)model compiler:(VBSMetalCompiler*)compiler windowBytes:(NSUInteger)bytes;
- (BOOL)runLinearTensor:(VBSTensor*)tensor input:(id<MTLBuffer>)input output:(id<MTLBuffer>)output error:(NSError**)error;
@end
NS_ASSUME_NONNULL_END
