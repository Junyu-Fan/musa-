#include <musa_runtime.h>
#include <musa_bf16.h>
template<class T> __device__ float fp(T x) { return (float)x; }
template<> __device__ float fp(__mt_bfloat16 x) { return __bfloat162float(x); }
template<class X,class R> __global__ void residual_fp32(
 const X* x,const R* r,float* y,int n,float scale,float inv,int mode) {
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
  float v=fp(r[i]);
  // Preserve a distinct FP32 scaling result before addition; no FMA.
  float scaled = mode==0 ? v/scale : v*inv;
  y[i]=fp(x[i])+scaled;
 }
}
extern "C" int esm_residual(const void* x,const void* r,float* y,
 int xd,int rd,int n,double scale,int mode,void* stream) {
 int b=(n+255)/256;if(b>4096)b=4096;if(!n)return 0;
 float s=(float)scale;
 float inv=mode==1 ? (float)(1.0/scale) : 1.0f/s;
 if(xd==0 && rd==0)residual_fp32<<<b,256,0,(musaStream_t)stream>>>((const float*)x,(const float*)r,y,n,s,inv,mode);
 else if(xd==0 && rd==1)residual_fp32<<<b,256,0,(musaStream_t)stream>>>((const float*)x,(const __mt_bfloat16*)r,y,n,s,inv,mode);
 else if(xd==1 && rd==0)residual_fp32<<<b,256,0,(musaStream_t)stream>>>((const __mt_bfloat16*)x,(const float*)r,y,n,s,inv,mode);
 else if(xd==1 && rd==1)residual_fp32<<<b,256,0,(musaStream_t)stream>>>((const __mt_bfloat16*)x,(const __mt_bfloat16*)r,y,n,s,inv,mode);
 else return -1;
 return (int)musaGetLastError();
}
