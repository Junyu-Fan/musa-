#include <musa_runtime.h>
#include <musa_bf16.h>
#include <math.h>
__global__ void fused_swiglu(const __mt_bfloat16* x,__mt_bfloat16* y,int n,int d,int mode){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  int j=(i/d)*(2*d)+i%d;
  float a=__bfloat162float(x[j]),b=__bfloat162float(x[j+d]);
  float e=(mode%2==0)?expf(-a):__expf(-a);
  float z=mode<2?a/(1.0f+e):a*(1.0f/(1.0f+e));
  __mt_bfloat16 rounded=__float2bfloat16_rn(z);
  y[i]=__float2bfloat16_rn(__bfloat162float(rounded)*b);
 }
}
extern "C" int esm_swiglu(const void*x,void*y,int n,int d,int mode,void*stream){
 if(n==0)return 0;if(n<0||d<=0||mode<0||mode>3)return -1;
 int blocks=(n+255)/256;if(blocks>4096)blocks=4096;
 fused_swiglu<<<blocks,256,0,(musaStream_t)stream>>>((const __mt_bfloat16*)x,(__mt_bfloat16*)y,n,d,mode);
 return (int)musaGetLastError();
}

__global__ void fused_swiglu_lut(const __mt_bfloat16* x,const __mt_bfloat16* table,__mt_bfloat16* y,int n,int d){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  int j=(i/d)*(2*d)+i%d;
  unsigned short bits=((const unsigned short*)x)[j];
  float activation=__bfloat162float(table[bits]);
  float branch=__bfloat162float(x[j+d]);
  y[i]=__float2bfloat16_rn(activation*branch);
 }
}
extern "C" int esm_swiglu_lut(const void*x,const void*table,void*y,int n,int d,void*stream){
 if(!n)return 0;if(n<0||d<=0)return -1;
 int blocks=(n+255)/256;if(blocks>4096)blocks=4096;
 fused_swiglu_lut<<<blocks,256,0,(musaStream_t)stream>>>((const __mt_bfloat16*)x,(const __mt_bfloat16*)table,(__mt_bfloat16*)y,n,d);
 return (int)musaGetLastError();
}
__global__ void fused_swiglu_lut_4096(const __mt_bfloat16* x,const __mt_bfloat16* table,__mt_bfloat16* y,int n){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
  const int d=4096;
  int j=(i/4096)*8192+(i%4096);
  unsigned short bits=((const unsigned short*)x)[j];
  float activation=__bfloat162float(table[bits]);
  float branch=__bfloat162float(x[j+d]);
  y[i]=__float2bfloat16_rn(activation*branch);
 }
}

extern "C" int esm_swiglu_lut_fast(const void*x,const void*table,void*y,int n,int d,void*stream){
 if(d!=4096)return esm_swiglu_lut(x,table,y,n,d,stream);
 if(!n)return 0;int b=(n+255)/256;if(b>4096)b=4096;
 fused_swiglu_lut_4096<<<b,256,0,(musaStream_t)stream>>>((const __mt_bfloat16*)x,(const __mt_bfloat16*)table,(__mt_bfloat16*)y,n);
 return (int)musaGetLastError();
}
