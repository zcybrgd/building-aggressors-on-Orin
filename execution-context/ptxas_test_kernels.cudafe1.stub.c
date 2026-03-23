#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "ptxas_test_kernels.fatbin.c"
extern void __device_stub__Z12simpleKernelPfi(float *, int);
extern void __device_stub__Z18sharedMemoryKernelPfS_i(float *, float *, int);
extern void __device_stub__Z19dynamicSharedKernelPfS_i(float *, float *, int);
extern void __device_stub__Z18highRegisterKernelPfS_i(float *, float *, int);
extern void __device_stub__Z20constantMemoryKernelPfS_i(float *, float *, int);
extern void __device_stub__Z11spillKernelPfS_i(float *, float *, int);
extern void __device_stub__Z22multipleBarriersKernelPfS_i(float *, float *, int);
extern void __device_stub__Z18staticGlobalKernelPfS_i(float *, float *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z12simpleKernelPfi(float *__par0, int __par1){__cudaLaunchPrologue(2);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaLaunch(((char *)((void ( *)(float *, int))simpleKernel)));}
# 5 "ptxas_test_kernels.cu"
void simpleKernel( float *__cuda_0,int __cuda_1)
# 5 "ptxas_test_kernels.cu"
{__device_stub__Z12simpleKernelPfi( __cuda_0,__cuda_1);




}
# 1 "ptxas_test_kernels.cudafe1.stub.c"
void __device_stub__Z18sharedMemoryKernelPfS_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))sharedMemoryKernel))); }
# 15 "ptxas_test_kernels.cu"
void sharedMemoryKernel( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 15 "ptxas_test_kernels.cu"
{__device_stub__Z18sharedMemoryKernelPfS_i( __cuda_0,__cuda_1,__cuda_2);
# 31 "ptxas_test_kernels.cu"
}
# 1 "ptxas_test_kernels.cudafe1.stub.c"
void __device_stub__Z19dynamicSharedKernelPfS_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))dynamicSharedKernel))); }
# 36 "ptxas_test_kernels.cu"
void dynamicSharedKernel( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 36 "ptxas_test_kernels.cu"
{__device_stub__Z19dynamicSharedKernelPfS_i( __cuda_0,__cuda_1,__cuda_2);
# 50 "ptxas_test_kernels.cu"
}
# 1 "ptxas_test_kernels.cudafe1.stub.c"
void __device_stub__Z18highRegisterKernelPfS_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))highRegisterKernel))); }
# 55 "ptxas_test_kernels.cu"
void highRegisterKernel( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 55 "ptxas_test_kernels.cu"
{__device_stub__Z18highRegisterKernelPfS_i( __cuda_0,__cuda_1,__cuda_2);
# 76 "ptxas_test_kernels.cu"
}
# 1 "ptxas_test_kernels.cudafe1.stub.c"
void __device_stub__Z20constantMemoryKernelPfS_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))constantMemoryKernel))); }
# 83 "ptxas_test_kernels.cu"
void constantMemoryKernel( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 83 "ptxas_test_kernels.cu"
{__device_stub__Z20constantMemoryKernelPfS_i( __cuda_0,__cuda_1,__cuda_2);
# 92 "ptxas_test_kernels.cu"
}
# 1 "ptxas_test_kernels.cudafe1.stub.c"
void __device_stub__Z11spillKernelPfS_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))spillKernel))); }
# 97 "ptxas_test_kernels.cu"
void spillKernel( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 97 "ptxas_test_kernels.cu"
{__device_stub__Z11spillKernelPfS_i( __cuda_0,__cuda_1,__cuda_2);
# 114 "ptxas_test_kernels.cu"
}
# 1 "ptxas_test_kernels.cudafe1.stub.c"
void __device_stub__Z22multipleBarriersKernelPfS_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))multipleBarriersKernel))); }
# 119 "ptxas_test_kernels.cu"
void multipleBarriersKernel( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 119 "ptxas_test_kernels.cu"
{__device_stub__Z22multipleBarriersKernelPfS_i( __cuda_0,__cuda_1,__cuda_2);
# 140 "ptxas_test_kernels.cu"
}
# 1 "ptxas_test_kernels.cudafe1.stub.c"
void __device_stub__Z18staticGlobalKernelPfS_i( float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(float *, float *, int))staticGlobalKernel))); }
# 147 "ptxas_test_kernels.cu"
void staticGlobalKernel( float *__cuda_0,float *__cuda_1,int __cuda_2)
# 147 "ptxas_test_kernels.cu"
{__device_stub__Z18staticGlobalKernelPfS_i( __cuda_0,__cuda_1,__cuda_2);
# 154 "ptxas_test_kernels.cu"
}
# 1 "ptxas_test_kernels.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T7) {  __nv_dummy_param_ref(__T7); __nv_save_fatbinhandle_for_managed_rt(__T7); __cudaRegisterEntry(__T7, ((void ( *)(float *, float *, int))staticGlobalKernel), _Z18staticGlobalKernelPfS_i, (-1)); __cudaRegisterEntry(__T7, ((void ( *)(float *, float *, int))multipleBarriersKernel), _Z22multipleBarriersKernelPfS_i, (-1)); __cudaRegisterEntry(__T7, ((void ( *)(float *, float *, int))spillKernel), _Z11spillKernelPfS_i, (-1)); __cudaRegisterEntry(__T7, ((void ( *)(float *, float *, int))constantMemoryKernel), _Z20constantMemoryKernelPfS_i, (-1)); __cudaRegisterEntry(__T7, ((void ( *)(float *, float *, int))highRegisterKernel), _Z18highRegisterKernelPfS_i, (-1)); __cudaRegisterEntry(__T7, ((void ( *)(float *, float *, int))dynamicSharedKernel), _Z19dynamicSharedKernelPfS_i, (-1)); __cudaRegisterEntry(__T7, ((void ( *)(float *, float *, int))sharedMemoryKernel), _Z18sharedMemoryKernelPfS_i, (-1)); __cudaRegisterEntry(__T7, ((void ( *)(float *, int))simpleKernel), _Z12simpleKernelPfi, (-1)); __cudaRegisterVariable(__T7, __shadow_var(constCoeffs,::constCoeffs), 0, 256UL, 1, 0); __cudaRegisterVariable(__T7, __shadow_var(globalBuffer,::globalBuffer), 0, 4096UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
