#import "VBSGGUF.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static NSString *const VBSErrorDomain = @"Verum.GGUF";
typedef NS_ENUM(uint32_t, VBSMetaType) { M_U8=0,M_I8=1,M_U16=2,M_I16=3,M_U32=4,M_I32=5,M_F32=6,M_BOOL=7,M_STRING=8,M_ARRAY=9,M_U64=10,M_I64=11,M_F64=12 };

@interface VBSGGUFMetadata ()
@property(nonatomic, readwrite) NSString *key; @property(nonatomic, readwrite) uint32_t type;
@property(nonatomic, readwrite) uint64_t valueOffset; @property(nonatomic, readwrite) uint64_t valueEnd;
@end
@implementation VBSGGUFMetadata @end

@interface VBSTensor ()
@property(nonatomic, readwrite) NSString *name; @property(nonatomic, readwrite) NSArray<NSNumber *> *shape;
@property(nonatomic, readwrite) VBSGGMLType type; @property(nonatomic, readwrite) uint64_t dataOffset;
@property(nonatomic, readwrite) uint64_t byteLength; @property(nonatomic, readwrite) uint64_t columns;
@property(nonatomic, readwrite) uint64_t rows; @property(nonatomic, readwrite) uint64_t packedRowBytes;
@end
@implementation VBSTensor @end

BOOL VBSGeometryForType(VBSGGMLType t, VBSBlockGeometry *g) {
    VBSBlockGeometry x = {0,0};
    switch (t) {
        case VBSGGMLTypeF32: x=(VBSBlockGeometry){1,4}; break;
        case VBSGGMLTypeF16: case VBSGGMLTypeBF16: x=(VBSBlockGeometry){1,2}; break;
        case VBSGGMLTypeI8: x=(VBSBlockGeometry){1,1}; break;
        case VBSGGMLTypeI16: x=(VBSBlockGeometry){1,2}; break;
        case VBSGGMLTypeI32: x=(VBSBlockGeometry){1,4}; break;
        case VBSGGMLTypeI64: case VBSGGMLTypeF64: x=(VBSBlockGeometry){1,8}; break;
        case VBSGGMLTypeQ4_0: x=(VBSBlockGeometry){32,18}; break;
        case VBSGGMLTypeQ4_1: x=(VBSBlockGeometry){32,20}; break;
        case VBSGGMLTypeQ5_0: x=(VBSBlockGeometry){32,22}; break;
        case VBSGGMLTypeQ5_1: x=(VBSBlockGeometry){32,24}; break;
        case VBSGGMLTypeQ8_0: x=(VBSBlockGeometry){32,34}; break;
        case VBSGGMLTypeQ8_1: x=(VBSBlockGeometry){32,40}; break;
        case VBSGGMLTypeQ2_K: x=(VBSBlockGeometry){256,84}; break;
        case VBSGGMLTypeQ3_K: x=(VBSBlockGeometry){256,110}; break;
        case VBSGGMLTypeQ4_K: x=(VBSBlockGeometry){256,144}; break;
        case VBSGGMLTypeQ5_K: x=(VBSBlockGeometry){256,176}; break;
        case VBSGGMLTypeQ6_K: x=(VBSBlockGeometry){256,210}; break;
        case VBSGGMLTypeQ8_K: x=(VBSBlockGeometry){256,292}; break;
        default: return NO;
    }
    if (g) *g=x; return YES;
}

typedef struct { const uint8_t *base; uint64_t size, off; } Cursor;
static BOOL take(Cursor *c, void *dst, uint64_t n) { if (n>c->size || c->off>c->size-n) return NO; if(dst) memcpy(dst,c->base+c->off,(size_t)n); c->off+=n; return YES; }
static BOOL u32(Cursor*c,uint32_t*v){return take(c,v,4);} static BOOL u64(Cursor*c,uint64_t*v){return take(c,v,8);}
static NSString *str(Cursor*c){uint64_t n=0;if(!u64(c,&n)||n>NSUIntegerMax||n>c->size-c->off)return nil;NSData*d=[NSData dataWithBytes:c->base+c->off length:(NSUInteger)n];c->off+=n;return [[NSString alloc]initWithData:d encoding:NSUTF8StringEncoding];}
static BOOL skipValue(Cursor *c,uint32_t t);
static BOOL skipArray(Cursor*c){uint32_t sub;uint64_t n;if(!u32(c,&sub)||!u64(c,&n))return NO;for(uint64_t i=0;i<n;i++)if(!skipValue(c,sub))return NO;return YES;}
static BOOL skipValue(Cursor*c,uint32_t t){uint64_t n=0;switch(t){case M_U8:case M_I8:case M_BOOL:n=1;break;case M_U16:case M_I16:n=2;break;case M_U32:case M_I32:case M_F32:n=4;break;case M_U64:case M_I64:case M_F64:n=8;break;case M_STRING:if(!u64(c,&n))return NO;break;case M_ARRAY:return skipArray(c);default:return NO;}return take(c,NULL,n);}
static NSError *err(NSString *s){return [NSError errorWithDomain:VBSErrorDomain code:1 userInfo:@{NSLocalizedDescriptionKey:s}];}

@interface VBSGGUFModel () { int _fd; void *_map; uint64_t _fileSize; }
@property(nonatomic, readwrite) NSString *path; @property(nonatomic, readwrite) uint32_t version,alignment;
@property(nonatomic, readwrite) NSDictionary<NSString *,VBSGGUFMetadata*> *metadata;
@property(nonatomic, readwrite) NSArray<VBSTensor*> *tensors;
@property(nonatomic, readwrite) NSDictionary<NSString *,VBSTensor*> *tensorsByName;
@end

@implementation VBSGGUFModel
- (instancetype)initWithPath:(NSString *)path error:(NSError **)error {
    if(!(self=[super init]))return nil; _fd=open(path.fileSystemRepresentation,O_RDONLY); if(_fd<0){if(error)*error=err(@"Cannot open GGUF");return nil;}
    struct stat st;if(fstat(_fd,&st)||st.st_size<24){if(error)*error=err(@"Invalid GGUF size");close(_fd);_fd=-1;return nil;}_fileSize=(uint64_t)st.st_size;
    _map=mmap(NULL,(size_t)_fileSize,PROT_READ,MAP_PRIVATE,_fd,0);if(_map==MAP_FAILED){if(error)*error=err(@"mmap failed");close(_fd);_fd=-1;return nil;}
    Cursor c={(const uint8_t*)_map,_fileSize,0};char magic[4];uint64_t nt=0,nm=0;
    if(!take(&c,magic,4)||memcmp(magic,"GGUF",4)||!u32(&c,&_version)||!u64(&c,&nt)||!u64(&c,&nm)){if(error)*error=err(@"Malformed GGUF header");return nil;}
    NSMutableDictionary *md=[NSMutableDictionary dictionaryWithCapacity:(NSUInteger)MIN(nm,UINT32_MAX)];
    for(uint64_t i=0;i<nm;i++){NSString*k=str(&c);uint32_t t;if(!k||!u32(&c,&t)){if(error)*error=err(@"Malformed metadata key");return nil;}VBSGGUFMetadata*e=[VBSGGUFMetadata new];e.key=k;e.type=t;e.valueOffset=c.off;if(!skipValue(&c,t)){if(error)*error=err([@"Malformed metadata: " stringByAppendingString:k]);return nil;}e.valueEnd=c.off;md[k]=e;}
    _metadata=[md copy]; NSNumber *a=[self numberForMetadataKey:@"general.alignment"];_alignment=a?a.unsignedIntValue:32;if(_alignment<8||(_alignment&(_alignment-1))){if(error)*error=err(@"Invalid GGUF alignment");return nil;}
    NSMutableArray *ts=[NSMutableArray arrayWithCapacity:(NSUInteger)MIN(nt,UINT32_MAX)];NSMutableDictionary*by=[NSMutableDictionary dictionary];
    for(uint64_t i=0;i<nt;i++){NSString*n=str(&c);uint32_t nd=0,type=0;uint64_t rel=0;if(!n||!u32(&c,&nd)||nd==0||nd>4){if(error)*error=err(@"Malformed tensor info");return nil;}NSMutableArray*shape=[NSMutableArray arrayWithCapacity:nd];uint64_t elements=1;for(uint32_t d=0;d<nd;d++){uint64_t x;if(!u64(&c,&x)||x==0||elements>UINT64_MAX/x){if(error)*error=err(@"Invalid tensor shape");return nil;}elements*=x;[shape addObject:@(x)];}if(!u32(&c,&type)||!u64(&c,&rel)){if(error)*error=err(@"Truncated tensor info");return nil;}VBSBlockGeometry g;if(!VBSGeometryForType((VBSGGMLType)type,&g)||shape[0].unsignedLongLongValue%g.elementsPerBlock){if(error)*error=err([NSString stringWithFormat:@"Unsupported or misaligned tensor encoding %u: %@",type,n]);return nil;}uint64_t cols=shape[0].unsignedLongLongValue,rows=elements/cols,blocks=cols/g.elementsPerBlock,rowBytes;if(blocks>UINT64_MAX/g.bytesPerBlock){if(error)*error=err(@"Tensor row size overflow");return nil;}rowBytes=blocks*g.bytesPerBlock;if(rows>UINT64_MAX/rowBytes){if(error)*error=err(@"Tensor byte size overflow");return nil;}VBSTensor*t=[VBSTensor new];t.name=n;t.shape=shape;t.type=(VBSGGMLType)type;t.dataOffset=rel;t.columns=cols;t.rows=rows;t.packedRowBytes=rowBytes;t.byteLength=rows*rowBytes;[ts addObject:t];by[n]=t;}
    if(c.off>UINT64_MAX-(_alignment-1)){if(error)*error=err(@"GGUF data offset overflow");return nil;}uint64_t aligned=((c.off+_alignment-1)/_alignment)*_alignment;for(VBSTensor*t in ts){if(t.dataOffset>UINT64_MAX-aligned){if(error)*error=err([@"Tensor offset overflow: " stringByAppendingString:t.name]);return nil;}uint64_t absolute=aligned+t.dataOffset;if(absolute>_fileSize||t.byteLength>_fileSize-absolute){if(error)*error=err([@"Tensor outside file: " stringByAppendingString:t.name]);return nil;}t.dataOffset=absolute;}
    _path=[path copy];_tensors=[ts copy];_tensorsByName=[by copy];return self;
}
- (void)dealloc { if(_map&&_map!=MAP_FAILED)munmap(_map,(size_t)_fileSize);if(_fd>=0)close(_fd); }
- (NSString*)stringForMetadataKey:(NSString*)key { VBSGGUFMetadata*e=_metadata[key];if(!e||e.type!=M_STRING)return nil;Cursor c={(const uint8_t*)_map,_fileSize,e.valueOffset};return str(&c); }
- (NSNumber*)numberForMetadataKey:(NSString*)key { VBSGGUFMetadata*e=_metadata[key];if(!e)return nil;const uint8_t*p=(const uint8_t*)_map+e.valueOffset;switch(e.type){case M_U8:return @(*(uint8_t*)p);case M_I8:return @(*(int8_t*)p);case M_U16:{uint16_t v;memcpy(&v,p,2);return @(v);}case M_I16:{int16_t v;memcpy(&v,p,2);return @(v);}case M_U32:{uint32_t v;memcpy(&v,p,4);return @(v);}case M_I32:{int32_t v;memcpy(&v,p,4);return @(v);}case M_U64:{uint64_t v;memcpy(&v,p,8);return @(v);}case M_I64:{int64_t v;memcpy(&v,p,8);return @(v);}case M_F32:{float v;memcpy(&v,p,4);return @(v);}case M_F64:{double v;memcpy(&v,p,8);return @(v);}case M_BOOL:return @(*p!=0);default:return nil;} }
- (const void*)bytesForTensor:(VBSTensor*)t { return (const uint8_t*)_map+t.dataOffset; }
- (BOOL)copyPackedTileOfTensor:(VBSTensor*)t rowStart:(uint64_t)rs rowCount:(uint64_t)rc columnStart:(uint64_t)cs columnCount:(uint64_t)cc destination:(void*)dst capacity:(NSUInteger)cap bytesCopied:(NSUInteger*)copied error:(NSError**)error { VBSBlockGeometry g;if(!dst||!VBSGeometryForType(t.type,&g)||cs%g.elementsPerBlock||cc%g.elementsPerBlock||rs>t.rows||rc>t.rows-rs||cs>t.columns||cc>t.columns-cs){if(error)*error=err(@"Tile is out of range or not quant-block aligned");return NO;}uint64_t tileRow=(cc/g.elementsPerBlock)*g.bytesPerBlock,total=tileRow*rc;if(total>cap||total>NSUIntegerMax){if(error)*error=err(@"Tile exceeds bounded staging window");return NO;}uint64_t colByte=(cs/g.elementsPerBlock)*g.bytesPerBlock;const uint8_t*src=(const uint8_t*)_map+t.dataOffset+rs*t.packedRowBytes+colByte;for(uint64_t r=0;r<rc;r++)memcpy((uint8_t*)dst+r*tileRow,src+r*t.packedRowBytes,(size_t)tileRow);if(copied)*copied=(NSUInteger)total;return YES; }
@end
