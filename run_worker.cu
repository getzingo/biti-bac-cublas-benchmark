// Sources:
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu
// https://github.com/hma02/cublasgemm-benchmark/blob/master/gemm.cu
// https://github.com/NVIDIA/cutlass/tree/main/tools/profiler

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) do { cudaError_t rc=(call); if(rc!=cudaSuccess) { \
  std::fprintf(stderr,"FATAL %s: %s\n",#call,cudaGetErrorString(rc)); std::exit(1); } } while(0)
#define BLAS_CHECK(call) do { cublasStatus_t rc=(call); if(rc!=CUBLAS_STATUS_SUCCESS) { \
  std::fprintf(stderr,"FATAL %s: %d\n",#call,int(rc)); std::exit(1); } } while(0)

static int number(const char *text, bool zero=false) {
  char *end; errno=0; long value=std::strtol(text,&end,10);
  if(errno || *end || end==text || value<(zero?0:1) || value>INT_MAX) {
    std::fprintf(stderr,"Invalid integer: %s\n",text); std::exit(2);
  }
  return int(value);
}
static long long now_ns() {
  timespec ts;
  if(clock_gettime(CLOCK_MONOTONIC_RAW,&ts)) { std::perror("clock_gettime"); std::exit(1); }
  return static_cast<long long>(ts.tv_sec)*1000000000LL+ts.tv_nsec;
}
int main(int argc,char **argv) {
  if(argc!=8) { std::fprintf(stderr,"Usage: run_worker wid m n k fp16|fp32|fp64 warmup runs\n"); return 2; }
  int wid=number(argv[1],true), m=number(argv[2]), n=number(argv[3]), k=number(argv[4]);
  int warmup=number(argv[6],true), runs=number(argv[7]);
  const bool fp16=!std::strcmp(argv[5],"fp16"), fp32=!std::strcmp(argv[5],"fp32");
  if(!fp16 && !fp32 && std::strcmp(argv[5],"fp64")) return 2;
  int dev, blas_version, runtime_version, driver_version;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  cublasHandle_t h; BLAS_CHECK(cublasCreate(&h));
  BLAS_CHECK(cublasGetVersion(h,&blas_version));
  BLAS_CHECK(cublasSetMathMode(h,fp16?CUBLAS_TENSOR_OP_MATH:CUBLAS_PEDANTIC_MATH));
  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st)); BLAS_CHECK(cublasSetStream(h,st));
  cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  size_t bytes=fp16?sizeof(__half):(fp32?sizeof(float):sizeof(double));

  if(m!=8192 || n!=8192 || k!=8192) { std::fprintf(stderr,"Only 8192 cubed supported\n"); return 2; }
  void *a,*b,*c;
  CUDA_CHECK(cudaMalloc(&a,size_t(m)*k*bytes));
  CUDA_CHECK(cudaMalloc(&b,size_t(k)*n*bytes));
  CUDA_CHECK(cudaMalloc(&c,size_t(m)*n*bytes));
  CUDA_CHECK(cudaMemset(a,0,size_t(m)*k*bytes));
  CUDA_CHECK(cudaMemset(b,0,size_t(k)*n*bytes));
  CUDA_CHECK(cudaMemset(c,0,size_t(m)*n*bytes));
  float alpha=1.f,beta=0.f; double da=1.,db=0.;
  auto gemm=[&]() {
    if(fp32) BLAS_CHECK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,
      static_cast<float*>(a),m,static_cast<float*>(b),k,&beta,static_cast<float*>(c),m));
    else if(fp16) BLAS_CHECK(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,
      a,CUDA_R_16F,m,b,CUDA_R_16F,k,&beta,c,CUDA_R_16F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else BLAS_CHECK(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&da,
      static_cast<double*>(a),m,static_cast<double*>(b),k,&db,static_cast<double*>(c),m));
  };
  std::printf("# META {\"protocol\":2,\"math_mode\":\"%s\",\"precision\":\"%s\",\"accumulation\":\"%s\",\"cublas_version\":%d,\"cuda_runtime\":%d,\"cuda_driver\":%d,\"sms\":%d,\"layout\":\"column-major\",\"transpose\":\"NN\",\"alpha\":1,\"beta\":0,\"input\":\"zero\"}\n",
      fp16?"tensor_op":"pedantic",argv[5],fp16||fp32?"fp32":"fp64",blas_version,runtime_version,driver_version,prop.multiProcessorCount);
  std::fprintf(stderr,"worker %d on %s (%d SMs)\n",wid,prop.name,prop.multiProcessorCount);
  for(int i=0;i<warmup;++i) gemm();
  CUDA_CHECK(cudaStreamSynchronize(st));
  std::puts("# READY"); std::fflush(stdout);
  char command[16];
  if(!std::fgets(command,sizeof(command),stdin) || std::strcmp(command,"GO\n")) {
    std::fprintf(stderr,"Missing parent start gate\n"); return 2;
  }
  for(int i=0;i<runs;++i) {
    long long begin=now_ns(); CUDA_CHECK(cudaEventRecord(e0,st));
    gemm(); CUDA_CHECK(cudaEventRecord(e1,st)); CUDA_CHECK(cudaStreamSynchronize(st));
    long long end=now_ns(); float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    std::printf("%d,%d,%.6f,%lld,%lld\n",wid,i,ms,begin,end); std::fflush(stdout);
  }
  CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
  BLAS_CHECK(cublasDestroy(h)); CUDA_CHECK(cudaStreamDestroy(st));
  CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c));
  std::fprintf(stderr,"worker %d done (%d runs)\n",wid,runs);
}
