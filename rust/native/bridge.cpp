// Narrow native-backend interop absent from tch 0.24. No model, I/O, or Python.
#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/autocast_mode.h>
#include <ATen/ops/_foreach_norm.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_fused_adamw.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#ifdef CAIRN_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include "routing_cubin.h"
#include <mutex>
static void driver_check(CUresult result) {
  const char* message=nullptr;
  if(result!=CUDA_SUCCESS) { cuGetErrorString(result,&message); TORCH_CHECK(false,"CUDA routing: ",message?message:"unknown error"); }
}
static void cairn_route_cuda(const int64_t* selected,int* counts,int* offsets,int64_t* totals,int64_t* routes,int n,int experts,cudaStream_t stream) {
  static std::once_flag initialized;
  static CUmodule module;
  static CUfunction count,prefix,pack;
  std::call_once(initialized,[] {
    driver_check(cuModuleLoadData(&module,cairn_routing_cubin));
    driver_check(cuModuleGetFunction(&count,module,"count_routes"));
    driver_check(cuModuleGetFunction(&prefix,module,"prefix_routes"));
    driver_check(cuModuleGetFunction(&pack,module,"pack_routes"));
  });
  int tiles=(n+255)/256;
  void* a[]={&selected,&counts,&n,&tiles};
  void* b[]={&counts,&offsets,&totals,&tiles,&experts};
  void* c[]={&selected,&offsets,&routes,&n,&tiles};
  driver_check(cuLaunchKernel(count,tiles,experts,1,256,1,1,0,stream,a,nullptr));
  driver_check(cuLaunchKernel(prefix,1,1,1,64,1,1,0,stream,b,nullptr));
  driver_check(cuLaunchKernel(pack,tiles,experts,1,256,1,1,0,stream,c,nullptr));
}
#endif
#include <string>
#include <vector>
// Preserve the sequential chunk recurrence and its matrix multiplications,
// but write each slice gradient once instead of allocating a full zero tensor
// for every select/slice backward. No approximation or parallel reassociation.
class CairnScan : public torch::autograd::Function<CairnScan> {
public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
      at::Tensor read,at::Tensor local,at::Tensor transition,at::Tensor write,at::Tensor initial) {
    const auto n=read.size(2);
    std::vector<at::Tensor> values, saved{read,transition};
    auto state=initial;
    for(int64_t c=0;c<n;++c) {
      saved.push_back(state);
      values.push_back(local.select(2,c)+read.select(2,c).matmul(state));
      state=transition.select(2,c).matmul(state)+write.select(2,c);
    }
    ctx->save_for_backward(saved);
    return {at::stack(values,2),state};
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list gradients) {
    const auto saved=ctx->get_saved_variables();
    const auto read=saved[0],transition=saved[1];
    const auto n=read.size(2);
    auto state_grad=gradients[1];
    // SelectBackward in the reference creates contiguous full-size gradients,
    // even when its forward input is a strided permutation. Preserve that
    // layout: downstream reductions can otherwise use a different sum order.
    auto grad_read=at::empty(read.sizes(),read.options());
    auto grad_transition=at::empty(transition.sizes(),transition.options());
    auto shape=saved[2].sizes().vec(); shape.insert(shape.begin()+2,n);
    auto grad_write=at::empty(shape,read.options());
    for(int64_t c=n-1;c>=0;--c) {
      const auto out=gradients[0].select(2,c);
      grad_read.select(2,c).copy_(out.matmul(saved[c+2].transpose(-1,-2)));
      grad_transition.select(2,c).copy_(state_grad.matmul(saved[c+2].transpose(-1,-2)));
      grad_write.select(2,c).copy_(state_grad);
      state_grad=transition.select(2,c).transpose(-1,-2).matmul(state_grad)
                   +read.select(2,c).transpose(-1,-2).matmul(out);
    }
    return {grad_read,gradients[0].contiguous(),grad_transition,grad_write,state_grad};
  }
};
static thread_local std::string error;
#define TRY try {
#define CATCH } catch(const std::exception& e) { error=e.what(); return -1; } return 0;
extern "C" {
int cairn_scan(at::Tensor* read,at::Tensor* local,at::Tensor* transition,at::Tensor* write,
               at::Tensor* initial,at::Tensor** outputs,at::Tensor** final) { TRY
  const auto result=CairnScan::apply(*read,*local,*transition,*write,*initial);
  *outputs=new at::Tensor(result[0]); *final=new at::Tensor(result[1]);
  CATCH
}
int cairn_memory(int reset, double* allocated, double* reserved) { TRY
#ifdef CAIRN_CUDA
  if(reset) c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  const auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  const auto index=static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
  *allocated=stats.allocated_bytes[index].peak;
  *reserved=stats.reserved_bytes[index].peak;
#else
  *allocated=0.; *reserved=0.;
#endif
  CATCH
}
int cairn_routes(at::Tensor* selected, int64_t experts, at::Tensor** routes_out, at::Tensor** counts_out) { TRY
#ifdef CAIRN_CUDA
  TORCH_CHECK(selected->is_cuda() && selected->scalar_type()==at::kLong && selected->dim()==2,
              "Routing requires CUDA int64 [tokens, top_k]");
  TORCH_CHECK(experts>=2 && experts<=64 && selected->numel()>0 && selected->numel()<=1048576,
              "Routing dimensions exceed supported bounds");
  const auto s=selected->contiguous();
  const int n=s.numel(), tiles=(n+255)/256;
  auto counts=at::empty({experts,tiles},s.options().dtype(at::kInt));
  auto offsets=at::empty_like(counts);
  auto totals=at::empty({experts},s.options());
  auto routes=at::empty({experts,n},s.options());
  cairn_route_cuda(s.data_ptr<int64_t>(),counts.data_ptr<int>(),offsets.data_ptr<int>(),
                   totals.data_ptr<int64_t>(),routes.data_ptr<int64_t>(),n,experts,
                   c10::cuda::getCurrentCUDAStream(s.get_device()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  *counts_out=new at::Tensor(totals.cpu()); // One host synchronization for all experts.
  *routes_out=new at::Tensor(routes);
#else
  TORCH_CHECK(false,"CUDA routing unavailable");
#endif
  CATCH
}
const char* cairn_error() { return error.c_str(); }
int cairn_settings() { TRY
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
  at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
  CATCH
}
int cairn_deterministic(int enabled) { TRY
  at::globalContext().setDeterministicAlgorithms(enabled,false);
  at::globalContext().setDeterministicCuDNN(enabled);
  at::globalContext().setBenchmarkCuDNN(false);
  CATCH
}
int cairn_autocast_enter(int enabled) {
  const int previous=at::autocast::is_autocast_enabled(at::kCUDA);
  at::autocast::set_autocast_enabled(at::kCUDA,enabled);
  at::autocast::increment_nesting(); return previous;
}
void cairn_autocast_exit(int previous) {
  if(at::autocast::decrement_nesting()==0) at::autocast::clear_cache();
  at::autocast::set_autocast_enabled(at::kCUDA,previous);
}
int cairn_normal(at::Tensor* t, uint64_t seed, double std) { TRY
  at::NoGradGuard guard;
  TORCH_CHECK(t->device().is_cpu(), "Paired initialization must occur on CPU");
  auto gen=at::make_generator<at::CPUGeneratorImpl>(seed);
  t->normal_(0., std, gen);
  CATCH
}
int cairn_clear_grad(at::Tensor* t) { TRY t->mutable_grad()=at::Tensor(); CATCH }
int cairn_backward(at::Tensor* out, at::Tensor* grad) { TRY
  torch::autograd::backward({*out}, {*grad}); CATCH
}
int cairn_adamw(at::Tensor** p, at::Tensor** m, at::Tensor** v, at::Tensor** s,
               size_t n, double lr, double wd) { TRY
  at::NoGradGuard guard;
  std::vector<at::Tensor> ps, gs, ms, vs, ss;
  for(size_t i=0;i<n;i++) {
    if(!p[i]->grad().defined()) continue;
    ps.push_back(*p[i]); gs.push_back(p[i]->grad());
    ms.push_back(*m[i]); vs.push_back(*v[i]); s[i]->add_(1.); ss.push_back(*s[i]);
  }
  if(!ps.empty()) at::_fused_adamw_(ps,gs,ms,vs,{},ss,lr,.9,.95,wd,1e-8,false,false);
  CATCH
}
int cairn_clip(at::Tensor** p, size_t n, double maximum, double* result) { TRY
  at::NoGradGuard guard;
  std::vector<at::Tensor> gs;
  for(size_t i=0;i<n;i++) if(p[i]->grad().defined()) gs.push_back(p[i]->grad());
  if(gs.empty()) { *result=0.; return 0; }
  auto norms=at::_foreach_norm(gs,2.);
  auto total=at::linalg_vector_norm(at::stack(norms),2.);
  *result=total.item<double>();
  TORCH_CHECK(std::isfinite(*result),"Nonfinite gradient norm");
  auto coefficient=(maximum/(total+1e-6)).clamp_max(1.);
  at::_foreach_mul_(gs,coefficient);
  CATCH
}
int cairn_rng_get(int cuda, at::Tensor** out) { TRY
#ifdef CAIRN_CUDA
  if(cuda) { *out=new at::Tensor(at::cuda::detail::getDefaultCUDAGenerator().get_state()); return 0; }
#endif
  TORCH_CHECK(!cuda,"CUDA unavailable");
  *out=new at::Tensor(at::detail::getDefaultCPUGenerator().get_state()); CATCH
}
int cairn_rng_set(int cuda, at::Tensor* state) { TRY
#ifdef CAIRN_CUDA
  if(cuda) { auto g=at::cuda::detail::getDefaultCUDAGenerator(); g.set_state(*state); return 0; }
#endif
  TORCH_CHECK(!cuda,"CUDA unavailable");
  auto g=at::detail::getDefaultCPUGenerator(); g.set_state(*state); CATCH
}
}
