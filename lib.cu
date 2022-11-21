#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <map>

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

void printMatrix(void *A, const char *prefix)
{
  __half hA[16];
  cudaMemcpy(hA, A, sizeof(__half) * 16, cudaMemcpyDeviceToHost);
  printf("%s", prefix);
  for (int i = 0; i < 16; i++)
  {
    printf("%.3f ", __half2float(hA[i]));
  }
  printf("\n\n");
}

// Everything assumes contiguous memory with stride == num_cols
struct Descriptor
{
  int num_rows;
  int num_cols;
  cudaDataType_t dtype;

  bool operator<(const Descriptor &other) const
  {
    return num_rows < other.num_rows && num_cols < other.num_cols && (int)dtype < (int)other.dtype;
  }
};

static std::map<Descriptor, cusparseLtMatDescriptor_t> dense_descriptors;
cusparseLtMatDescriptor_t *dense_desc_get_or_init(cusparseLtHandle_t *handle, Descriptor desc)
{
  auto it = dense_descriptors.find(desc);
  if (it != dense_descriptors.end())
    return &dense_descriptors[desc];
  else
  {
    cusparseLtMatDescriptor_t mat;
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(handle, &mat, desc.num_rows, desc.num_cols, desc.num_cols, 16, desc.dtype, CUSPARSE_ORDER_ROW))
    dense_descriptors[desc] = mat;
    return &dense_descriptors[desc];
  }
}

static std::map<Descriptor, cusparseLtMatDescriptor_t> structured_descriptors;
cusparseLtMatDescriptor_t *structured_desc_get_or_init(cusparseLtHandle_t *handle, Descriptor desc)
{
  auto it = structured_descriptors.find(desc);
  if (it != structured_descriptors.end())
    return &structured_descriptors[desc];
  else
  {
    cusparseLtMatDescriptor_t mat;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(handle, &mat, desc.num_rows, desc.num_cols, desc.num_cols,
                                                      16, desc.dtype, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT))
    structured_descriptors[desc] = mat;
    return &structured_descriptors[desc];
  }
}

struct MatmulDescriptor
{
  cusparseOperation_t opA;
  cusparseOperation_t opB;
  Descriptor descA;
  Descriptor descB;
  Descriptor descC;
  cusparseLtMatDescriptor_t *matA;
  cusparseLtMatDescriptor_t *matB;
  cusparseLtMatDescriptor_t *matC;
  cusparseComputeType compute_type;
  bool operator<(const MatmulDescriptor &other) const {
    return (int)opA < (int)other.opA && (int)opB < (int)other.opB && descA < other.descA &&
      descB < other.descB && descC < other.descC && (int)compute_type < (int)other.compute_type
  }
};
static std::map<MatmulDescriptor, cusparseLtMatmulDescriptor_t> matmul_descriptors;
cusparseLtMatmulDescriptor_t *matmul_desc_get_or_init(cusparseLtHandle_t *handle, MatmulDescriptor desc)
{
  auto it = matmul_descriptors.find(desc);
  if (it != matmul_descriptors.end())
    return &matmul_descriptors[desc];
  else
  {
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(handle, &matmul, desc.opA, desc.opB, desc.matA, desc.matB, desc.matC,
                                                  desc.matC, desc.compute_type))
    matmul_descriptors[desc] = matmul;
    return &matmul_descriptors[desc];
  }
}

extern "C" void sparse_matmul(void *context, void *A, void *B, void *C, int num_A_rows, int num_A_cols, int num_B_cols)
{
  cusparseLtHandle_t handle = ((Context *)context)->cslt_handle;

  Descriptor descA{num_A_rows, num_A_cols, CUDA_R_16F};
  cusparseLtMatDescriptor_t *matA = structured_desc_get_or_init(&handle, descA);
  Descriptor descB{num_A_cols, num_B_cols, CUDA_R_16F};
  cusparseLtMatDescriptor_t *matB = dense_desc_get_or_init(&handle, descB);
  Descriptor descC{num_A_rows, num_B_cols, CUDA_R_16F};
  cusparseLtMatDescriptor_t *matC = dense_desc_get_or_init(&handle, descC);

  cusparseLtMatmulDescriptor_t *matmul = matmul_desc_get_or_init(&handle, MatmulDescriptor{
    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, descA, descB, descC, matA, matB, matC, CUSPARSE_COMPUTE_16F
  });

  CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, matmul, A, A, CUSPARSELT_PRUNE_SPMMA_STRIP, nullptr))
  // int d_valid;
  // CHECK_CUDA(cudaMalloc((void **)&d_valid, sizeof(d_valid)));
  CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, matmul, A, nullptr, nullptr))

  __half *A_compressed;
  size_t compressed_size;
  CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&handle, matA, &compressed_size))
  CHECK_CUDA(cudaMalloc(&A_compressed, compressed_size))
  CHECK_CUSPARSE(cusparseLtSpMMACompress2(&handle, matA, 1, CUSPARSE_OPERATION_NON_TRANSPOSE, A, A_compressed, nullptr))

  int alg = 0;
  cusparseLtMatmulPlan_t plan;
  cusparseLtMatmulAlgSelection_t alg_sel;
  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
  CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(&handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, matmul, &alg_sel, 0))
  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, A_compressed, B, &beta, C, C, nullptr, nullptr, 0))
  CHECK_CUDA(cudaFree(A_compressed))
}