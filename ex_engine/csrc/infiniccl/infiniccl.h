/* InfiniCCL public API — minimal subset for allreduce via dlopen */
#ifndef INFINICCL_H_
#define INFINICCL_H_
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif

#define INFINICCL_UNIQUE_ID_BYTES 128

typedef void *infinicclComm_t;
typedef struct { char internal[INFINICCL_UNIQUE_ID_BYTES]; } infinicclUniqueId;

typedef enum { INFINICCL_SUCCESS = 0, INFINICCL_ERROR = 1 } infinicclResult_t;
typedef enum { infinicclFloat16 = 8, infinicclFloat32 = 10 } infinicclDataType_t;
typedef enum { infinicclSum = 0 } infinicclRedOp_t;

infinicclResult_t infinicclGetUniqueId(infinicclUniqueId *id);
infinicclResult_t infinicclCommInitRank(infinicclComm_t *comm, int nranks,
                                        infinicclUniqueId id, int rank);
infinicclResult_t infinicclCommDestroy(infinicclComm_t comm);
infinicclResult_t infinicclAllReduce(const void *sendbuff, void *recvbuff,
                                     size_t count, infinicclDataType_t datatype,
                                     infinicclRedOp_t op, infinicclComm_t comm,
                                     void *stream);

#ifdef __cplusplus
}
#endif
#endif
