#import <Foundation/Foundation.h>
@class VBSGGUFModel, VBSTensor;

NS_ASSUME_NONNULL_BEGIN
typedef NS_ENUM(NSUInteger, VBSOpKind) {
    VBSOpEmbedding, VBSOpLinear, VBSOpNormalize, VBSOpAttention,
    VBSOpRotary, VBSOpActivation, VBSOpElementwise, VBSOpStateSpace,
    VBSOpConvolution, VBSOpCache, VBSOpProjection
};

@interface VBSNode : NSObject
@property(nonatomic) NSString *identifier;
@property(nonatomic) VBSOpKind kind;
@property(nonatomic) NSArray<NSString *> *inputs;
@property(nonatomic) NSArray<NSString *> *outputs;
@property(nonatomic, nullable) VBSTensor *weight;
@property(nonatomic) NSDictionary<NSString *, id> *attributes;
@end

@interface VBSGraph : NSObject
@property(nonatomic) NSArray<VBSNode *> *nodes;
@property(nonatomic) NSDictionary<NSString *, id> *facts;
@property(nonatomic) NSString *fingerprint;
@end

@interface VBSGraphCompiler : NSObject
- (nullable VBSGraph *)compileModel:(VBSGGUFModel *)model error:(NSError **)error;
@end
NS_ASSUME_NONNULL_END

