#pragma once

#include "core/op_algo.h"
#include "nccl.h"

namespace ixformer::comm {

const uint8_t MAX_TENSOR_NDIM = 8;
constexpr size_t DEFAULT_SHM_SIZE = 16 * 1024 * 1024 * sizeof(float);

struct Comm;
typedef Comm *Comm_t;


struct TensorDesc {
    void *data_ptr;
    ncclDataType_t dtype;
    uint64_t numel;
    uint8_t ndim;
    int64_t shape[MAX_TENSOR_NDIM];
    int64_t stride[MAX_TENSOR_NDIM];
    bool contiguous;
};


/**
 * @brief Generate unique communicator id.
 *
 * Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling ncclCommInitRank.
 *
 * @param commId: the unique id of communicator, it is created in main rank, and broadcast other rank.
 */
void getUniqueId(ncclUniqueId *commId);

/**
 * @brief Serialize commId to string.
 * @param commId: the unique id of communicator.
 * @return: serialized string.
 */
std::string serializeUniqueId(const ncclUniqueId &commId);

/**
 * @brief Deserialize the string of commId.
 * @param commIdStr: serialized string by serializeUniqueId.
 * @param commId: output commId
 */
void deserializeUniqueId(const std::string &commIdStr, ncclUniqueId *commId);

/**
 * @brief Creates a new communicator (multi process version).
 *
 * Rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a CUDA device, which has to be set before calling ncclCommInitRank.
 *
 * It is important to ensure that the current process's CUDA device is set by cudaSetDevice before calling this function,
 * otherwise, an exception will be thrown.
 *
 * @param comm: Communicator
 * @param nranks: the number of ranks.
 * @param commId: the unique id of communicator.
 * @param rank: the rank of current process
 * @param shm_size: Unlike NCCL, IxFormer communication relies on CUDA IPC for communication by shared memory.
 *        If the shm_size is not provided, it will use the default value: DEFAULT_SHM_SIZE.
 * @throw CommError: Throw CommError when an error is encountered.
 */
void initRank(Comm_t *comm, int nranks, ncclUniqueId commId, int rank, size_t shm_size = DEFAULT_SHM_SIZE);

/**
 * @brief Finalize a communicator.
 *
 * ncclCommFinalize flushes all issued communications,
 * and marks communicator state as ncclInProgress. The state will change to ncclSuccess
 * when the communicator is globally quiescent and related resources are freed; then,
 * calling ncclCommDestroy can locally free the rest of the resources (e.g. communicator
 * itself) without blocking.
 *
 * @param comm: Communicator
 * @throw CommError: Throw CommError when an error is encountered.
 */
void destroy(Comm_t comm);

void delete_comm_resuouces(Comm_t comm);

/**
 * @brief Whether is initiated.
 * @param comm: Communicator
 */
bool isInitiated(Comm_t comm);

/**
 * @brief Gets ncclComm_t.
 * @param comm: Communicator
 */
ncclComm_t getNcclComm(Comm_t comm);

/**
 * @brief Gets the number of ranks in the communicator clique
 * @param comm: Communicator
 */
int getWorldSize(Comm_t comm);

/**
 * @brief Gets the number of nodes in the communicator clique
 * @param comm: Communicator
 */
int getNumNodes(Comm_t comm);

/**
 * @brief Returns the user-ordered "rank" associated with the communicator.
 * @param comm: Communicator
 */
int getRank(Comm_t comm);

/**
 * @brief Returns the cuda device number associated with the communicator.
 * @param comm: Communicator
 */
int getDevice(Comm_t comm);

/**
 * @brief Gets shared memory size in the communicator clique
 * @param comm: Communicator
 */
uint64_t getIpcShmSize(Comm_t comm);


// ============================================================================
// Collective communication operations
//
// Collective communication operations must be called separately for each
// communicator in a communicator clique.
//
// They return when operations have been enqueued on the CUDA stream.
//
// Since they may perform inter-CPU synchronization, each call has to be done
// from a different thread or process, or need to use Group Semantics (see
// below).
// ============================================================================

/**
 * @brief Barrier the member of the communicator
 * @param comm: Communicator
 * @param stream: CUDA Stream
 */
void barrier(Comm_t comm, cudaStream_t stream);

/**
 * @brief All-Gather
 *
 * Each device gathers sendcount values from other GPUs into senddata,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvdata
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if senddata == recvdata + rank * sendcount.
 *
 * @param comm: Communicator
 * @param senddata: send data
 * @param recvdata: recv data
 * @param sendcount: the number of send elements, it is not nbytes.
 * @param dtype: data type
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void allGather(Comm_t comm, const void *senddata, void *recvdata, size_t sendcount, ncclDataType_t dtype,
               cudaStream_t stream, AllGatherAlgo algo = AllGatherAlgo::kNone);

/**
 * @brief All-Reduce
 *
 * Reduces data arrays of length count in senddata using op operation, and
 * leaves identical copies of result on each recvdata.
 *
 * In-place operation will happen if senddata == recvdata.
 *
 * @param comm: Communicator
 * @param senddata: send data
 * @param recvdata: recv data
 * @param count: the number of elements, it is not nbytes.
 * @param dtype: data type
 * @param op：Reduce type: ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void allReduce(Comm_t comm, const void *senddata, void *recvdata, size_t count, ncclDataType_t dtype,
               ncclRedOp_t op, cudaStream_t stream, AllReduceAlgo algo = AllReduceAlgo::kNone);

/**
 * @brief Whether is supported non-contiguous tensors
 * @param comm: Communicator
 * @param dtype: data type
 * @param shape: tensor shape
 * @param ndim: the ndim of tensor
 * @param numel: the number of tensor
 * @param op: Reduce type: ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg
 * @return: supported
 */
bool allReduceStrideSupported(Comm_t comm, ncclDataType_t dtype, const int64_t *shape, int ndim, uint64_t numel, ncclRedOp_t op);

/**
 * @brief All-Reduce for non-contiguous tensors
 * @param comm: Communicator
 * @param senddata: send tensor
 * @param recvdata: recv tensor
 * @param stream: CUDA Stream
 */
void allReduceStride(Comm_t comm, const TensorDesc &senddata, TensorDesc &recvdata, cudaStream_t stream);

/**
 * @brief Send data from senddata to rank peer.
 *
 * Rank peer needs to call ncclRecv with the same datatype and the same count from this
 * rank. This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
 * need to progress concurrently to complete.
 *
 * @param comm: Communicator
 * @param senddata: send data
 * @param count: the number of send elements, it is not nbytes.
 * @param dtype: data type
 * @param peer: the destination rank
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void send(Comm_t comm, const void *senddata, size_t count, ncclDataType_t dtype, int peer, cudaStream_t stream,
          SendAlgo algo = SendAlgo::kNone);

/**
 * @brief Receive data from rank peer into recvdata.
 *
 * Rank peer needs to call ncclSend with the same datatype and the same count to this
 * rank. This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
 * need to progress concurrently to complete.
 *
 * @param comm: Communicator
 * @param recvdata: recv data
 * @param count: the number of recv elements, it is not nbytes.
 * @param dtype: data type
 * @param peer: source rank
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void recv(Comm_t comm, void *recvdata, size_t count, ncclDataType_t dtype, int peer, cudaStream_t stream,
          RecvAlgo algo = RecvAlgo::kNone);

/**
 * @brief Reduces data arrays of length count in senddata into recvdata using op operation.
 *
 * Recvdata may be NULL on all calls except for root device.
 * root is the rank (not the CUDA device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if senddata == recvdata.
 *
 * @param comm: Communicator
 * @param senddata: send data
 * @param recvdata: recv data
 * @param count:  the number of elements, it is not nbytes.
 * @param dtype: data type
 * @param op: Reduce type: ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg
 * @param root: root rank
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void reduce(Comm_t comm, const void *senddata, void *recvdata, size_t count, ncclDataType_t dtype, ncclRedOp_t op,
            int root, cudaStream_t stream, ReduceAlgo algo = ReduceAlgo::kNone);

/**
 * @brief Broadcast
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if senddata == recvdata.
 *
 * @param comm: Communicator
 * @param senddata: send data
 * @param recvdata: recv data
 * @param count: the number of elements, it is not nbytes.
 * @param dtype: data type
 * @param root: root rank
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void broadcast(Comm_t comm, const void *senddata, void *recvdata, size_t count, ncclDataType_t dtype, int root,
               cudaStream_t stream, BroadcastAlgo algo = BroadcastAlgo::kNone);

/**
 *
 * @brief Reduce-Scatter
 *
 * Reduces data in senddata using op operation and leaves reduced result
 * scattered over the devices so that recvdata on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that senddata
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvdata == senddata + rank * recvcount.
 *
 * @param comm: Communicator
 * @param senddata: send data
 * @param recvdata: recv data
 * @param recvcount: the number of recv elements, it is not nbytes.
 * @param dtype: data type
 * @param op：Reduce type: ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void reduceScatter(Comm_t comm, const void *senddata, void *recvdata,
                   size_t recvcount, ncclDataType_t dtype, ncclRedOp_t op, cudaStream_t stream,
                   ReduceScatterAlgo algo = ReduceScatterAlgo::kNone);

/**
 * @brief Send data from src_rank to dst_rank on src_rank process, recv data on dst_rank.
 *
 * @param comm: Communicator
 * @param data: send data to dst rank if current rank is src_rank, recv data if current rank is dst_rank.
 * @param count: the number of send/recv elements, it is not nbytes.
 * @param dtype: data type
 * @param src_rank: src rank
 * @param dst_rank: dst rank
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void p2p(Comm_t comm, void *data, size_t count, ncclDataType_t dtype, int src_rank, int dst_rank, cudaStream_t stream,
         SendAlgo algo = SendAlgo::kNone);

/**
 * @brief The all ranks of communicator send senddata to dst_rank.
 *
 * @param comm: Communicator
 * @param senddata: send data
 * @param recvdatas: recv datas，it is two-dim array, shape: [WorldSize, RecvDataPointer],
 *        the first dim is host pointer，the second dim is GPU pointer,
 *        it can be nullptr when current rank is not dst rank.
 * @param sendcount: the number of send elements, it is not nbytes.
 * @param dst_rank: dst rank
 * @param dtype: data type
 * @param stream: CUDA Stream
 * @param algo: Algorithm
 * @throw CommError: Throw CommError when an error is encountered.
 */
void gather(Comm_t comm, const void *senddata, void **recvdatas, size_t sendcount, int dst_rank, ncclDataType_t dtype,
            cudaStream_t stream, GatherAlgo algo = GatherAlgo::kNone);


}// namespace ixformer::comm
