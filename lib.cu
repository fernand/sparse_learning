#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cusparseLt.h>

#define CHECK_CUDA(func)                                         \
  {                                                              \
    cudaError_t status = (func);                                 \
    if (status != cudaSuccess)                                   \
    {                                                            \
      printf("CUDA API failed at line %d with error: %s (%d)\n", \
             __LINE__, cudaGetErrorString(status), status);      \
    }                                                            \
  }

#define CHECK_CUSPARSE(func)                                         \
  {                                                                  \
    cusparseStatus_t status = (func);                                \
    if (status != CUSPARSE_STATUS_SUCCESS)                           \
    {                                                                \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
             __LINE__, cusparseGetErrorString(status), status);      \
    }                                                                \
  }

struct Context
{
  cusparseLtHandle_t cslt_handle;
  Context()
  {
    cusparseLtInit(&cslt_handle);
  }
};

extern "C" Context *get_context()
{
  return new Context();
}

void printMatrix(void *A, const char *prefix) {
  __half hA[16];
  cudaMemcpy(hA, A, sizeof(__half)*16, cudaMemcpyDeviceToHost);
  printf("%s", prefix);
  for (int i=0; i<16; i++) {
    printf("%.3f ", __half2float(hA[i]));
  }
  printf("\n\n");
}

extern "C" void sparse_matmul(void *context, void *A, void *B, void *C, int num_rows, int num_cols)
{
  printMatrix(A, "A gpu ");
  cusparseLtHandle_t handle = static_cast<Context*>(context)->cslt_handle;

  cusparseLtMatDescriptor_t matA, matB, matC;
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, num_rows, num_cols, num_cols, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, num_rows, num_cols, num_rows, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, num_rows, num_cols, num_rows, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW))

  cusparseLtMatmulDescriptor_t matmul;
  CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, &matA, &matB, &matC, &matC, CUSPARSE_COMPUTE_16F))

  // Prune A in place
  CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, A, A, CUSPARSELT_PRUNE_SPMMA_STRIP, nullptr))
  int d_valid;
  CHECK_CUDA(cudaMalloc((void **)&d_valid, sizeof(d_valid)));
  CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, A, &d_valid, nullptr))
  printMatrix(A, "A pruned ");

  // Create a new compressed A
  __half *A_compressed;
  size_t compressed_size;
  CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&handle, &matA, &compressed_size))
  CHECK_CUDA(cudaMalloc(&A_compressed, compressed_size))
  CHECK_CUSPARSE(cusparseLtSpMMACompress2(&handle, &matA, 1, CUSPARSE_OPERATION_NON_TRANSPOSE, A, A_compressed, nullptr))
  printMatrix(A_compressed, "A compressed ");

  // Find the best kernel
  int alg = 0;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;
  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
  CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, 0))
  float alpha = 1.0f;
  float beta = 1.0f;
  // CHECK_CUSPARSE(cusparseLtMatmulSearch(&handle, &plan, &alpha, A_compressed, B, &beta, C, C, nullptr, nullptr, 0));
  // int alg_id;
  // CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id)))
  // int32_t splitK, splitKBuffers;
  // cusparseLtSplitKMode_t splitKMode;
  // CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K, &splitK, sizeof(splitK)))
  // CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_MODE, &splitKMode, sizeof(splitKMode)))
  // CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_BUFFERS, &splitKBuffers, sizeof(splitKBuffers)))
  // printf("alg_id: %i, splitK: %i, splitKBuffers: %i, splitkMode: %i\n", alg_id, (int)splitK, (int)splitKBuffers, splitKMode);

  size_t workspace_size;
  CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size))
  void *d_workspace = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&d_workspace, workspace_size))
  printf("worksparse size: %i\n", (int)workspace_size);
  // Perform the matrix multiplication
  CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, A_compressed, B, &beta, C, C, d_workspace, nullptr, 0))
}