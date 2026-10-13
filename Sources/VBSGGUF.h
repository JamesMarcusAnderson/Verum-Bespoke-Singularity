#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(uint32_t, VBSGGMLType) {
    VBSGGMLTypeF32 = 0, VBSGGMLTypeF16 = 1,
    VBSGGMLTypeQ4_0 = 2, VBSGGMLTypeQ4_1 = 3,
    VBSGGMLTypeQ5_0 = 6, VBSGGMLTypeQ5_1 = 7,
    VBSGGMLTypeQ8_0 = 8, VBSGGMLTypeQ8_1 = 9,
    VBSGGMLTypeQ2_K = 10, VBSGGMLTypeQ3_K = 11,
    VBSGGMLTypeQ4_K = 12, VBSGGMLTypeQ5_K = 13,
    VBSGGMLTypeQ6_K = 14, VBSGGMLTypeQ8_K = 15,
    VBSGGMLTypeIQ2_XXS = 16, VBSGGMLTypeIQ2_XS = 17,
    VBSGGMLTypeIQ3_XXS = 18, VBSGGMLTypeIQ1_S = 19,
    VBSGGMLTypeIQ4_NL = 20, VBSGGMLTypeIQ3_S = 21,
    VBSGGMLTypeIQ2_S = 22, VBSGGMLTypeIQ4_XS = 23,
    VBSGGMLTypeI8 = 24, VBSGGMLTypeI16 = 25,
    VBSGGMLTypeI32 = 26, VBSGGMLTypeI64 = 27,
    VBSGGMLTypeF64 = 28, VBSGGMLTypeIQ1_M = 29,
    VBSGGMLTypeBF16 = 30, VBSGGMLTypeTQ1_0 = 34,
    VBSGGMLTypeTQ2_0 = 35, VBSGGMLTypeMXFP4 = 39,
};

typedef struct { uint32_t elementsPerBlock; uint32_t bytesPerBlock; } VBSBlockGeometry;
FOUNDATION_EXPORT BOOL VBSGeometryForType(VBSGGMLType type, VBSBlockGeometry *outGeometry);

@interface VBSGGUFMetadata : NSObject
@property(nonatomic, readonly) NSString *key;
@property(nonatomic, readonly) uint32_t type;
@property(nonatomic, readonly) uint64_t valueOffset;
@property(nonatomic, readonly) uint64_t valueEnd;
@end

@interface VBSTensor : NSObject
@property(nonatomic, readonly) NSString *name;
@property(nonatomic, readonly) NSArray<NSNumber *> *shape;
@property(nonatomic, readonly) VBSGGMLType type;
@property(nonatomic, readonly) uint64_t dataOffset;
@property(nonatomic, readonly) uint64_t byteLength;
@property(nonatomic, readonly) uint64_t columns;
@property(nonatomic, readonly) uint64_t rows;
@property(nonatomic, readonly) uint64_t packedRowBytes;
@end

@interface VBSGGUFModel : NSObject
@property(nonatomic, readonly) NSString *path;
@property(nonatomic, readonly) uint32_t version;
@property(nonatomic, readonly) uint32_t alignment;
@property(nonatomic, readonly) NSDictionary<NSString *, VBSGGUFMetadata *> *metadata;
@property(nonatomic, readonly) NSArray<VBSTensor *> *tensors;
@property(nonatomic, readonly) NSDictionary<NSString *, VBSTensor *> *tensorsByName;
- (nullable instancetype)initWithPath:(NSString *)path error:(NSError **)error;
- (nullable NSString *)stringForMetadataKey:(NSString *)key;
- (nullable NSNumber *)numberForMetadataKey:(NSString *)key;
- (const void *)bytesForTensor:(VBSTensor *)tensor;
- (BOOL)copyPackedTileOfTensor:(VBSTensor *)tensor
                       rowStart:(uint64_t)rowStart rowCount:(uint64_t)rowCount
                    columnStart:(uint64_t)columnStart columnCount:(uint64_t)columnCount
                     destination:(void *)destination capacity:(NSUInteger)capacity
                     bytesCopied:(NSUInteger *)bytesCopied error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END

