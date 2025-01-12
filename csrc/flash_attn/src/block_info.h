/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 定义了一个名为 BlockInfo 的结构体模板，用于在CUDA设备（GPU）上计算和管理与序列长度相关的信息。
///        该结构体主要用于处理变长序列（variable-length sequences）的情况，
///        特别是在处理注意力机制（如Transformer中的自注意力机制）时，序列长度可能会变化。
/// @tparam Varlen 指示是否处理变长序列
template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb]))
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    const int sum_s_q;         /* 表示Query序列的累积长度。如果Varlen为false，或没有提供累积q序列长度，则其值为-1，否则为根据bidb取对应的值 */
    const int sum_s_k;         /* 表示Value序列的累积长度。 */
    const int actual_seqlen_q; /* 表示当前块的query序列的实际长度。如果Varlen为false，则直接使用params.seqlen_q；否则，为cu_seqlens_q[bidb + 1]-sum_s_q */
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int seqlen_k_cache;  /* 表示当前块的key序列的长度 */
    const int actual_seqlen_k; /* 表示当前key序列的实际长度 */
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
