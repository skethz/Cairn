// Build-only tool. The model executable does not link or invoke NVRTC.
#include <nvrtc.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
int main(int argc,char** argv) {
  if(argc!=3) return 2;
  std::ifstream in(argv[1]);
  const std::string source((std::istreambuf_iterator<char>(in)),{});
  if(source.empty() || source.size()>1024*1024) return 3;
  nvrtcProgram program;
  if(nvrtcCreateProgram(&program,source.c_str(),"cairn_routing.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS) return 4;
  const char* options[]={"--gpu-architecture=sm_100","--std=c++17","--generate-line-info"};
  const auto result=nvrtcCompileProgram(program,3,options);
  size_t n=0;
  nvrtcGetProgramLogSize(program,&n);
  std::vector<char> log(n);
  nvrtcGetProgramLog(program,log.data());
  std::cerr<<log.data();
  if(result!=NVRTC_SUCCESS) return 5;
  if(nvrtcGetCUBINSize(program,&n)!=NVRTC_SUCCESS) return 6;
  std::vector<unsigned char> bytes(n);
  if(nvrtcGetCUBIN(program,reinterpret_cast<char*>(bytes.data()))!=NVRTC_SUCCESS) return 7;
  std::ofstream out(argv[2]);
  out<<"static const unsigned char cairn_routing_cubin[]={";
  for(const auto byte:bytes) out<<static_cast<unsigned>(byte)<<',';
  out<<"};\n";
  nvrtcDestroyProgram(&program);
  return out.good()?0:8;
}
