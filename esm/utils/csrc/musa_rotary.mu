// Preserve the Torch path's multiply-round, multiply-round, add-round semantics.
// In particular, do not compile with fast-math or contract the operations to FMA.
#include <musa_runtime.h>
#include <musa_bf16.h>

template <class T> __device__ float as_float(T x) { return (float)x; }
template <> __device__ float as_float(__mt_bfloat16 x) {
  return __bfloat162float(x);
}
template <class T> __device__ T rounded(float x) { return (T)x; }
template <> __device__ __mt_bfloat16 rounded(float x) {
  return __float2bfloat16_rn(x);
}

// Q/K are contiguous [B,L,H,D]; cos and sign-folded sin are [L,1,D].
template <class T>
__global__ void rotary_pair(const T* q, const T* k, const T* cs,
                            const T* sn, T* oq, T* ok,
                            int n, int L, int H, int D) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    const int d = i % D;
    const int partner = i + (d < D / 2 ? D / 2 : -D / 2);
    const int c = ((i / (H * D)) % L) * D + d;
    const float a = as_float(cs[c]), b = as_float(sn[c]);
    const T qa = rounded<T>(as_float(q[i]) * a);
    const T qb = rounded<T>(as_float(q[partner]) * b);
    const T ka = rounded<T>(as_float(k[i]) * a);
    const T kb = rounded<T>(as_float(k[partner]) * b);
    oq[i] = rounded<T>(as_float(qa) + as_float(qb));
    ok[i] = rounded<T>(as_float(ka) + as_float(kb));
  }
}

extern "C" int esm_rotary_abi_version() { return 1; }

extern "C" int esm_rotary_pair(const void* q, const void* k,
                                const void* c, const void* s,
                                void* oq, void* ok,
                                int dtype, int n, int L, int H, int D,
                                void* stream) {
  if (n <= 0 || n > (1 << 30) || L <= 0 || H <= 0 || D <= 0 || D % 2)
    return -1;
  const int blocks = (n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096;
  if (dtype == 0) {
    rotary_pair<<<blocks, 256, 0, (musaStream_t)stream>>>(
        (const __mt_bfloat16*)q, (const __mt_bfloat16*)k,
        (const __mt_bfloat16*)c, (const __mt_bfloat16*)s,
        (__mt_bfloat16*)oq, (__mt_bfloat16*)ok, n, L, H, D);
  } else if (dtype == 1) {
    rotary_pair<<<blocks, 256, 0, (musaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)c, (const float*)s,
        (float*)oq, (float*)ok, n, L, H, D);
  } else {
    return -1;
  }
  return (int)musaGetLastError();
}
