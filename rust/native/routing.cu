// Build-time NVRTC input: integer-only, stable row-major MoE dispatch.
// No host headers, runtime compiler, floating-point math, or atomics.
typedef long long int64_t;
constexpr int tile=256;
extern "C" __global__ void count_routes(const int64_t* selected,int* counts,int n,int tiles) {
  __shared__ int sum[tile];
  const int lane=threadIdx.x,pos=blockIdx.x*tile+lane;
  sum[lane]=pos<n && selected[pos]==blockIdx.y;
  __syncthreads();
  for(int stride=128;stride;stride/=2) {
    if(lane<stride) sum[lane]+=sum[lane+stride];
    __syncthreads();
  }
  if(lane==0) counts[blockIdx.y*tiles+blockIdx.x]=sum[0];
}
extern "C" __global__ void prefix_routes(const int* counts,int* offsets,int64_t* totals,int tiles,int experts) {
  const int expert=threadIdx.x;
  if(expert>=experts) return;
  int total=0;
  for(int i=0;i<tiles;++i) {
    offsets[expert*tiles+i]=total;
    total+=counts[expert*tiles+i];
  }
  totals[expert]=total;
}
extern "C" __global__ void pack_routes(const int64_t* selected,const int* offsets,int64_t* routes,int n,int tiles) {
  __shared__ int scan[tile];
  const int lane=threadIdx.x,pos=blockIdx.x*tile+lane;
  const int flag=pos<n && selected[pos]==blockIdx.y;
  scan[lane]=flag;
  __syncthreads();
  for(int stride=1;stride<tile;stride*=2) {
    const int previous=lane>=stride ? scan[lane-stride] : 0;
    __syncthreads();
    scan[lane]+=previous;
    __syncthreads();
  }
  if(flag) routes[static_cast<int64_t>(blockIdx.y)*n+offsets[blockIdx.y*tiles+blockIdx.x]+scan[lane]-1]=pos;
}
