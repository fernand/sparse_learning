#include <stdlib.h>
#include <stdio.h>
#include <cusparseLt.h>
// #include <cuda_bf16.h>
// https://docs.nvidia.com/cuda/cusparselt
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSELt/spmma/spmma_example.cpp

const int64_t num_rows = 512;
const int64_t num_cols = 512;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main() {
  __half* dA_host = (__half*)malloc(sizeof(__half)*num_rows*num_cols);
  // __nv_bfloat16 blah;
  for (int i=0; i<num_rows*num_cols; i++)
    dA_host[i] = (__half)((float)rand()/(float)RAND_MAX);
  for (int i=512*512-16; i<512*512; i++)
    printf("%f ", __half2float(dA_host[i]));
  printf("\n\n");
  __half* dA;
  CHECK_CUDA(cudaMalloc(&dA, sizeof(__half)*num_rows*num_cols));
  CHECK_CUDA(cudaMemcpy(dA, dA_host, sizeof(__half)*num_rows*num_cols, cudaMemcpyHostToDevice));

  cusparseLtHandle_t handle;
  cusparseLtMatmulDescriptor_t matmul;
  cudaStream_t stream = nullptr;
  cusparseLtMatDescriptor_t matA, matB, matC;

  CHECK_CUSPARSE(cusparseLtInit(&handle));

  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, num_rows, num_cols, num_cols, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, num_rows, num_cols, num_rows, 16, CUDA_R_16F, CUSPARSE_ORDER_COL));
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, num_rows, num_cols, num_rows, 16, CUDA_R_16F, CUSPARSE_ORDER_COL));

  CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &matA, &matB, &matC, &matC, CUSPARSE_COMPUTE_16F));

  CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE, stream));
  int d_valid;
  CHECK_CUDA(cudaMalloc((void**)&d_valid, sizeof(d_valid)));
  CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, &d_valid, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(dA_host, dA, sizeof(__half)*num_rows*num_cols, cudaMemcpyDeviceToHost));
  for (int i=512*512-16; i<512*512; i++)
    printf("%f ", __half2float(dA_host[i]));
  printf("\n\n");

  __half* dA_compressed;
  size_t compressed_size; //294912
  CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&handle, &matA, &compressed_size));
  CHECK_CUDA(cudaMalloc(&dA_compressed, compressed_size));

  CHECK_CUSPARSE(cusparseLtSpMMACompress2(&handle, &matA, 1, CUSPARSE_OPERATION_NON_TRANSPOSE, dA, dA_compressed, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  printf("%i\n", (int)compressed_size);
  __half* dA_compressed_host = (__half*)malloc(compressed_size);
  CHECK_CUDA(cudaMemcpy(dA_compressed_host, dA_compressed, compressed_size, cudaMemcpyDeviceToHost));
  for (int i=131072-32; i < 131072; i++)
    printf("%f ", __half2float(dA_compressed_host[i]));
  printf("\n\n");

  for (int i=131072; i < 131072+32; i++)
    printf("%f ", __half2float(dA_compressed_host[i]));
  printf("\n");

  // cusparseLtMatmulPlan_t plan;
  // cusparseLtMatmulAlgSelection_t alg_sel;
  // CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
  // int alg = 0;
  // CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)));
  // CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, 0));

  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matB);
  cusparseLtMatDescriptorDestroy(&matC);
  // cusparseLtMatmulPlanDestroy(&plan);
  cusparseLtDestroy(&handle);
}
