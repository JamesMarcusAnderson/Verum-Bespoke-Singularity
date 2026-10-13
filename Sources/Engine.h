#import <Foundation/Foundation.h>
NS_ASSUME_NONNULL_BEGIN
@interface Engine : NSObject
- (nullable instancetype)initWithModelPath:(NSString*)path windowMiB:(NSUInteger)windowMiB error:(NSError**)error;
- (void)printInspection;
- (BOOL)compileDiscoveredHome:(NSError**)error;
@end
NS_ASSUME_NONNULL_END

