#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
NS_ASSUME_NONNULL_BEGIN
@interface VBSFabric : NSObject
@property(nonatomic, readonly) NSArray<id<MTLDevice>> *devices;
- (instancetype)initWithDevices:(NSArray<id<MTLDevice>>*)devices;
- (nullable id<MTLBuffer>)viewOrCopyBuffer:(id<MTLBuffer>)source
                                  toDevice:(id<MTLDevice>)destination
                                      wait:(BOOL)wait error:(NSError**)error;
@end
NS_ASSUME_NONNULL_END

