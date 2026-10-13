#import "VBSMetal.h"
#import "VBSGraph.h"

static NSString *const MetalError=@"Verum.Metal";
static NSError *merr(NSString*s){return [NSError errorWithDomain:MetalError code:1 userInfo:@{NSLocalizedDescriptionKey:s}];}

static NSString *quantPrelude(void) { return
@"#include <metal_stdlib>\nusing namespace metal;\n"
@"struct q40{half d;uchar q[16];};struct q41{half d;half m;uchar q[16];};"
@"struct q50{half d;uchar h[4];uchar q[16];};struct q51{half d;half m;uchar h[4];uchar q[16];};"
@"struct q80{half d;char q[32];};struct q81{float d;float s;char q[32];};"
@"struct q2k{uchar s[16];uchar q[64];half d;half dm;};struct q3k{uchar h[32];uchar q[64];uchar s[12];half d;};"
@"struct q4k{half d;half dm;uchar s[12];uchar q[128];};struct q5k{half d;half dm;uchar s[12];uchar h[32];uchar q[128];};"
@"struct q6k{uchar ql[128];uchar qh[64];char s[16];half d;};struct q8k{float d;char q[256];short sums[16];};\n"
@"inline float bf(ushort x){return as_type<float>(uint(x)<<16);}\n"
@"inline uint2 sm(device const uchar*s,uint g){if(g<4)return uint2(s[g]&63,s[g+4]&63);return uint2((s[g+4]&15)|((s[g-4]>>6)<<4),(s[g+4]>>4)|((s[g]>>6)<<4));}\n"
@"inline int s3(device const uchar*s,uint g){uint a0=uint(s[0])|(uint(s[1])<<8)|(uint(s[2])<<16)|(uint(s[3])<<24);uint a1=uint(s[4])|(uint(s[5])<<8)|(uint(s[6])<<16)|(uint(s[7])<<24);uint a2=uint(s[8])|(uint(s[9])<<8)|(uint(s[10])<<16)|(uint(s[11])<<24);uint u[4];u[0]=(a0&0x0f0f0f0f)|(((a2>>0)&0x03030303)<<4);u[1]=(a1&0x0f0f0f0f)|(((a2>>2)&0x03030303)<<4);u[2]=((a0>>4)&0x0f0f0f0f)|(((a2>>4)&0x03030303)<<4);u[3]=((a1>>4)&0x0f0f0f0f)|(((a2>>6)&0x03030303)<<4);return int((u[g>>2]>>(8*(g&3)))&255);}\n";
}

static NSString *decoder(VBSGGMLType t, uint32_t *block, NSString **structName) {
    switch(t){
        case VBSGGMLTypeQ4_0:*block=32;*structName=@"q40";return @"uint p=i&15,q=i<16?(b.q[p]&15):(b.q[p]>>4);return float(b.d)*float(int(q)-8);";
        case VBSGGMLTypeQ4_1:*block=32;*structName=@"q41";return @"uint p=i&15,q=i<16?(b.q[p]&15):(b.q[p]>>4);return float(b.d)*float(q)+float(b.m);";
        case VBSGGMLTypeQ5_0:*block=32;*structName=@"q50";return @"uint p=i&15,l=i<16?(b.q[p]&15):(b.q[p]>>4),h=(uint(b.h[i>>3])>>(i&7))&1;return float(b.d)*float(int(l|(h<<4))-16);";
        case VBSGGMLTypeQ5_1:*block=32;*structName=@"q51";return @"uint p=i&15,l=i<16?(b.q[p]&15):(b.q[p]>>4),h=(uint(b.h[i>>3])>>(i&7))&1;return float(b.d)*float(l|(h<<4))+float(b.m);";
        case VBSGGMLTypeQ8_0:*block=32;*structName=@"q80";return @"return float(b.d)*float(b.q[i]);";
        case VBSGGMLTypeQ8_1:*block=32;*structName=@"q81";return @"return b.d*float(b.q[i])+b.s;";
        case VBSGGMLTypeQ2_K:*block=256;*structName=@"q2k";return @"uint h=i>>7,w=i&127,p=w>>5,l=w&31,si=h*8+p*2+(l>>4),q=(uint(b.q[h*32+l])>>(2*p))&3,ps=b.s[si];return float(b.d)*float(ps&15)*float(q)-float(b.dm)*float(ps>>4);";
        case VBSGGMLTypeQ3_K:*block=256;*structName=@"q3k";return @"uint h=i>>7,w=i&127,p=w>>5,l=w&31,si=h*8+p*2+(l>>4),mask=1u<<(p+4*h);int q=int((uint(b.q[h*32+l])>>(2*p))&3)-((uint(b.h[l])&mask)?0:4);return float(b.d)*float(s3(b.s,si)-32)*float(q);";
        case VBSGGMLTypeQ4_K:*block=256;*structName=@"q4k";return @"uint g=i>>5,l=i&31;uint2 x=sm(b.s,g);uint q=(uint(b.q[(g>>1)*32+l])>>((g&1)*4))&15;return float(b.d)*float(x.x)*float(q)-float(b.dm)*float(x.y);";
        case VBSGGMLTypeQ5_K:*block=256;*structName=@"q5k";return @"uint g=i>>5,l=i&31;uint2 x=sm(b.s,g);uint q=(uint(b.q[(g>>1)*32+l])>>((g&1)*4))&15;q|=((uint(b.h[l])>>g)&1)<<4;return float(b.d)*float(x.x)*float(q)-float(b.dm)*float(x.y);";
        case VBSGGMLTypeQ6_K:*block=256;*structName=@"q6k";return @"uint h=i>>7,w=i&127,p=w>>5,l=w&31,li=h*64+(p&1)*32+l,lo=(uint(b.ql[li])>>(p<2?0:4))&15,hi=(uint(b.qh[h*32+l])>>(2*p))&3,si=h*8+p*2+(l>>4);return float(b.d)*float(b.s[si])*float(int(lo|(hi<<4))-32);";
        case VBSGGMLTypeQ8_K:*block=256;*structName=@"q8k";return @"return b.d*float(b.q[i]);";
        default:return nil;
    }
}

static NSString *linearSource(VBSGGMLType t) {
    NSString*load=nil;
    if(t==VBSGGMLTypeF32)load=@"device const float*w=(device const float*)raw;v=w[r*n+i];";
    else if(t==VBSGGMLTypeF16)load=@"device const half*w=(device const half*)raw;v=float(w[r*n+i]);";
    else if(t==VBSGGMLTypeBF16)load=@"device const ushort*w=(device const ushort*)raw;v=bf(w[r*n+i]);";
    else {uint32_t bs=0;NSString*sn=nil,*body=decoder(t,&bs,&sn);if(!body)return nil;load=[NSString stringWithFormat:@"device const %@*w=(device const %@*)raw;device const %@&b=w[r*(n/%u)+i/%u];v=dq(b,i%%%u);",sn,sn,sn,bs,bs,bs];return [NSString stringWithFormat:@"%@ inline float dq(device const %@&b,uint i){%@}\nkernel void linear_tile(device const uchar*raw[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],constant uint&rows[[buffer(4)]],constant uint&add[[buffer(5)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;for(uint i=0;i<n;i++){float v;%@a+=v*x[i];}if(add)y[r]+=a;else y[r]=a;}\n",quantPrelude(),sn,body,load];}
    return [NSString stringWithFormat:@"%@ kernel void linear_tile(device const uchar*raw[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],constant uint&rows[[buffer(4)]],constant uint&add[[buffer(5)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;for(uint i=0;i<n;i++){float v;%@a+=v*x[i];}if(add)y[r]+=a;else y[r]=a;}\n",quantPrelude(),load];
}

static NSString *primitiveSource(void){return @"#include <metal_stdlib>\nusing namespace metal;\n"
@"kernel void normalize(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],constant float&eps[[buffer(4)]],uint i[[thread_position_in_grid]]){if(i>=n)return;float ss=0;for(uint j=0;j<n;j++)ss+=x[j]*x[j];y[i]=x[i]*rsqrt(ss/float(n)+eps)*w[i];}\n"
@"kernel void activation(device const float*x[[buffer(0)]],device float*y[[buffer(1)]],constant uint&n[[buffer(2)]],uint i[[thread_position_in_grid]]){if(i<n)y[i]=x[i]/(1+exp(-x[i]));}\n"
@"kernel void elementwise(device const float*a[[buffer(0)]],device const float*b[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<n)y[i]=a[i]+b[i];}\n"
@"kernel void embedding(device const float*w[[buffer(0)]],device float*y[[buffer(1)]],constant uint&token[[buffer(2)]],constant uint&dim[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<dim)y[i]=w[ulong(token)*dim+i];}\n"
@"kernel void rotary(device float*x[[buffer(0)]],constant uint&pos[[buffer(1)]],constant uint&headDim[[buffer(2)]],constant float&theta[[buffer(3)]],uint i[[thread_position_in_grid]]){uint pair=i*2,within=pair%headDim;if(within+1>=headDim)return;float a=x[pair],b=x[pair+1],ang=float(pos)/pow(theta,float(within)/float(headDim));x[pair]=a*cos(ang)-b*sin(ang);x[pair+1]=a*sin(ang)+b*cos(ang);}\n"
@"kernel void attention(device const float*q[[buffer(0)]],device const float*k[[buffer(1)]],device const float*v[[buffer(2)]],device float*out[[buffer(3)]],constant uint&seq[[buffer(4)]],constant uint&heads[[buffer(5)]],constant uint&hd[[buffer(6)]],uint h[[thread_position_in_grid]]){if(h>=heads)return;for(uint d=0;d<hd;d++){float top=-INFINITY,sum=0,acc=0;for(uint t=0;t<seq;t++){float s=0;for(uint j=0;j<hd;j++)s+=q[h*hd+j]*k[(t*heads+h)*hd+j];s*=rsqrt(float(hd));float nt=max(top,s),a=exp(top-nt),b=exp(s-nt);acc=acc*a+v[(t*heads+h)*hd+d]*b;sum=sum*a+b;top=nt;}out[h*hd+d]=acc/sum;}}\n"
@"kernel void convolution(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],constant uint&kw[[buffer(4)]],uint i[[thread_position_in_grid]]){if(i>=n)return;float a=0;for(uint k=0;k<kw;k++)if(i>=k)a+=x[i-k]*w[k];y[i]=a;}\n"
@"kernel void state_space(device const float*x[[buffer(0)]],device const float*a[[buffer(1)]],device const float*b[[buffer(2)]],device float*state[[buffer(3)]],device float*y[[buffer(4)]],constant uint&n[[buffer(5)]],uint i[[thread_position_in_grid]]){if(i<n){state[i]=a[i]*state[i]+b[i]*x[i];y[i]=state[i];}}\n"
@"kernel void cache_store(device const float*x[[buffer(0)]],device float*cache[[buffer(1)]],constant uint&offset[[buffer(2)]],constant uint&n[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<n)cache[offset+i]=x[i];}\n";}

@interface VBSMetalCompiler(){NSMutableDictionary<NSString*,id<MTLComputePipelineState>>*_cache;}@end
@implementation VBSMetalCompiler
- (instancetype)initWithDevice:(id<MTLDevice>)d{if((self=[super init])){_device=d;_queue=[d newCommandQueue];_cache=[NSMutableDictionary dictionary];}return self;}
- (id<MTLComputePipelineState>)compile:(NSString*)src function:(NSString*)fn key:(NSString*)key error:(NSError**)error{id p=_cache[key];if(p)return p;MTLCompileOptions*o=[MTLCompileOptions new];o.fastMathEnabled=YES;id<MTLLibrary>l=[_device newLibraryWithSource:src options:o error:error];if(!l)return nil;id<MTLFunction>f=[l newFunctionWithName:fn];if(!f){if(error)*error=merr([@"Generated MSL omitted " stringByAppendingString:fn]);return nil;}p=[_device newComputePipelineStateWithFunction:f error:error];if(p)_cache[key]=p;return p;}
- (id<MTLComputePipelineState>)linearPipelineForType:(VBSGGMLType)t error:(NSError**)error{NSString*s=linearSource(t);if(!s){if(error)*error=merr([NSString stringWithFormat:@"No proven MSL decoder for GGML type %u",t]);return nil;}return [self compile:s function:@"linear_tile" key:[NSString stringWithFormat:@"linear:%u",t] error:error];}
- (id<MTLComputePipelineState>)primitivePipelineForNode:(VBSNode*)n error:(NSError**)error{NSString*fn=nil;switch(n.kind){case VBSOpEmbedding:fn=@"embedding";break;case VBSOpNormalize:fn=@"normalize";break;case VBSOpAttention:fn=@"attention";break;case VBSOpRotary:fn=@"rotary";break;case VBSOpActivation:fn=@"activation";break;case VBSOpElementwise:fn=@"elementwise";break;case VBSOpStateSpace:fn=@"state_space";break;case VBSOpConvolution:fn=@"convolution";break;case VBSOpCache:fn=@"cache_store";break;default:break;}if(!fn){if(error)*error=merr([NSString stringWithFormat:@"No primitive for inferred operation %lu",(unsigned long)n.kind]);return nil;}return [self compile:primitiveSource() function:fn key:[@"primitive:" stringByAppendingString:fn] error:error];}
@end
