/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 用于存储与 QKV（Query, Key, Value）矩阵相关的参数
struct Qkv_params {
    using index_t = int64_t; /* 一个类型别名，定义为 int64_t，用于表示索引或步长（stride）等整数值。 */
    // The QKV matrices.
    void *__restrict__ q_ptr; /* 指向Query Tensor的指针，__restrict__ 是一个编译器优化提示，表示这些指针指向的内存区域不会重叠 */
    void *__restrict__ k_ptr; /* 指向Key Tensor的指针 */
    void *__restrict__ v_ptr; /* 指向Value Tensor的指针 */

    // The stride between rows of the Q, K and V matrices.
    /**
     * 多维数组在内存中相邻元素之间的偏移量
     * q/k/v_batch_stride为QKV的batch维度的步长
     * q/k/v_row_stride为QKV在row维度的步长
     * q/k/v_head_stride为QKV在head维度的步长
     * */
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.MHA/MQA/GQA中的head数量
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 继承自Qkv_params，包含Qkv_params参数的同时扩展了forward计算相关的参数。
struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void *__restrict__ o_ptr;      /* 指向输出矩阵的指针 */
    void *__restrict__ oaccum_ptr; /* 指向累加输出矩阵的指针 */

    // The stride between rows of O.
    /* 定义了输出矩阵O在内存中的布局，主要是batch、row以及head等各维度的stride */
    index_t o_batch_stride; /* 输出的batch跨步 */
    index_t o_row_stride;   /* 输出o中每行跨步 */
    index_t o_head_stride;  /* 输出o中的每个head之间的跨步 */

    // The pointer to the P matrix.
    void *__restrict__ p_ptr; /* attention中的注意力分数矩阵 */

    // The pointer to the softmax sum.
    void *__restrict__ softmax_lse_ptr;      /* 指向softmax lse结果的指针 */
    void *__restrict__ softmax_lseaccum_ptr; /* 指向累积softmax LSE结果的指针 */

    // The dimensions.
    /**
     * 分别表示batch size，q的序列长度、k的序列长度、新的 Key 序列长度（可能用于增量解码）、hidden size、
     * 内存对齐之后qk的序列长度以及hidden size大小，旋转位置编码的维度、总的Query数量
     **/
    int b /* batch size */, seqlen_q /* q的序列长度 */, seqlen_k /* k的序列长度 */, seqlen_knew /* 新k的长度 */, d /* head dim */, seqlen_q_rounded /* 求整后的q长度 */, seqlen_k_rounded /* 求整后的k长度 */, d_rounded /* 求整后的head size */, rotary_dim /* rope的维度 */, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;      /* softmax的缩放因子 */
    float scale_softmax_log2; /* softmax的对数缩放因子 */

    // array of length b+1 holding starting offset of each sequence.
    int *__restrict__ cu_seqlens_q; /* 长度为b+1的数组，存储每个序列的起始偏移地址。存储query的累积序列长度 */
    int *__restrict__ cu_seqlens_k; /* 存储Key的累积序列长度 */

    // If provided, the actual length of each k sequence.
    int *__restrict__ seqused_k; /* 表示每个k序列的实际长度 */

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void *__restrict__ knew_ptr;
    void *__restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding. 旋转embedding的cos与sin矩阵
    void *__restrict__ rotary_cos_ptr;
    void *__restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache. kv cache的batch索引
    int *__restrict__ cache_batch_idx;

    // Paged KV cache
    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).dropout中的p值，概率值
    float p_dropout; /* dropout的概率值p */
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t; /* uint8_t类型的p值 */

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    float softcap;

    // Random state. pytorch的API，用于存储 Philox 伪随机数生成器的状态PhiloxState， cuda-graph安全的
    at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t *rng_state;

    bool is_bf16;   /* 是否为bf16数据类型 */
    bool is_causal; /* 是否为因果推断 */

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative; /* 是否使用累积序列长度。如果为true，seqlen_k = cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb]。*/

    bool is_rotary_interleaved; /* 是否使用交错的旋转位置编码 */

    int num_splits; // For split-KV version

    void *__restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;            // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped; // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
