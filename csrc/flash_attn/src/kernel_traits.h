/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

/// @brief 定义了FlashAttention内核的基本特性，包含了数据类型、内存布局、原子操作等等。
/// @tparam elem_type 数据类型，默认为cutlass::half_t，可选项为bfloat16_t
/// @tparam kHeadDim_ attention的head dimension
/// @tparam kBlockM_ M维度上block的维度
/// @tparam kBlockN_ N维度上block的维度
/// @tparam kNWarps_ 每个线程块中使用warp数量
template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t>
struct Flash_kernel_traits {
    /**
     * 如果 CUDA 架构 >= 800(如 A100)，使用 elem_type 作为数据类型，并启用 cp.async(异步拷贝)功能。
     * 否则，默认使用 cutlass::half_t，并禁用 cp.async。
     **/
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type; /* 数据类型，可能是 elem_type 或 cutlass::half_t。 */
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float; /* 累加器数据类型，固定为 float。 */
    using index_t = int64_t;    /* 索引类型，固定为 int64_t。 */

    /**
     * 矩阵乘法原子操作(MMA)的实现，根据CUDA架构选择不同的实现
     * 对于 SM80 及以上架构，
     * 如果数据类型为half_t则使用 SM80_16x8x16_F32F16F16F32_TN；
     * 如果数据类型为bfloat_t，则使用 SM80_16x8x16_F32BF16BF16F32_TN
     **/
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>; /* 对于 SM75 架构，使用 SM75_16x8x8_F32F16F16F32_TN。 */
#endif

    /**
     * SmemCopyAtom与SmemCopyAtomTransposed：共享内存的拷贝原子操作，根据cuda框架选择不同的实现
     **/
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
/// @brief 继承自 Flash_kernel_traits，进一步定义了前向传播(forward pass)的 Flash Attention 内核特性。
/// @tparam elem_type 数据类型，默认为cutlass::half_t，可选项为bfloat16_t
/// @tparam Base 基类，即 Flash_kernel_traits。
/// @tparam kHeadDim_ attention的head dimension
/// @tparam kBlockM_ M维度上block的大小
/// @tparam kBlockN_ N维度上block的大小
/// @tparam kNWarps_ 每个线程块中使用warp数量
/// @tparam Is_Q_in_regs_ 是否将 Q (查询矩阵)存储在寄存器中。
/// @tparam Share_Q_K_smem_ 是否共享 Q 和 K (键矩阵)的共享内存。
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;       /* 每个线程块中的 warp 数量。 */
    static constexpr int kNThreads = kNWarps * 32; /* 每个线程块中的线程数量，等于 kNWarps * 32。 */

    static constexpr int kBlockM = kBlockM_;                                                       /* M 维度上的分块大小。 */
    static constexpr int kBlockN = kBlockN_;                                                       /* N 维度上的分块大小。 */
    static constexpr int kHeadDim = kHeadDim_;                                                     /* 注意力 head dimension，目前支持的值有32,64,96,128,160,192,224,256 */
    static_assert(kHeadDim % 32 == 0);                                                             /* 静态断言kHeadDim的维度可以被32整除 */
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;                               /* K 维度上的分块大小，用于共享内存，如果HeadDim可以被64整除，则为64否则为32 */
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32); /* K 维度上的分块大小，用于全局内存,如果可以被128整除则为128，否则看是否可以被64整除用64，都不行则选择32 */
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;                                     /* 如果kBlockKSmem为32，则kSwizzle的值为2，否则为3 */

    /**
     * 声明TiledMMA，定义了一个CTA如何协同处理shape为(MNK)的矩阵乘法，里面有4个warps，即128个线程
     * thread block循环可以处理MNK为(64, 16, 16)的矩阵乘法
     */
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>, // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;  /* Tile的大小为(64, 16, 16) */
    // using TiledMma = decltype(
    //     make_tiled_mma(typename Base::MMA_Atom_Arch,
    //     Layout<Shape<Int<kNWarps>, _1, _1>>{}, Tile<Int<16 * kNWarps>, _16, _16>{})
    // );

    /* 定义qkv矩阵在共享内存中的atom布局。shape为(8 * kBlockKSmem), stride为(kBlockKSmem,1)，同时对数据的Layout进行Swizzle操作。 */
    /* Swizzle 是一种内存访问优化技术，通过重新排列数据在内存中的存储顺序，减少 Bank Conflict（存储体冲突），从而提高内存访问效率。 */
    using SmemLayoutAtomQ = decltype(                           /* decltype用于推导出表达式的类型，将推导出的类型赋值给SmemLayoutAtomQ */
        composition(Swizzle<kSwizzle, 3, 3>{},                  /* composition将Swizzle对象与Layout对象组合在一起。一个Swizzle对象，用于对数据进行位操作，拥有优化共享内存的访问模式，kSiwzzle为掩码位数， */
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,         /* Layout的shape为8*32或8*64 */
                           Stride<Int<kBlockKSmem>, _1>>{}));   /* 一个Layout对象，定义了数据的shape与stride。根据定义可以看出是row-major存储 */
    
    /* 生成Q矩阵的共享内存tile，shape为gQ一样，LayoutAtom为SmemLayoutAtomQ */
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{})); /* shape为kBlockM, kHeadDim(Br, d) */

    /* K 和 V 矩阵的共享内存布局。 */
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{})); /* shape为kBlockN, kHeadDim(Bc, d) */

    // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
    /**
     * 定义矩阵V转置后的共享内存布局
     * SmemLayoutKV是KV的共享内存布局
     * make_layout创建一个shape为(kHeadDim, kBlockN)，行优化的布局
     **/
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    /* 输出矩阵的共享内存布局 */
    using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                                 Layout<Shape<_8, Int<kBlockKSmem>>,
                                                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    /* 输出矩阵的共享内存的拷贝原子 */
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    /* Q与KV矩阵的共享内存大小 */
    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    /* 总的共享内存大小。如果QK共享内存，则为二者的最大值，否则为二者之和 */
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;

    /* 每次全局内存加载的元素数量，向量化加载 */
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    /**
     * 每行每次全局内存加载的线程数量
     * */
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    /**
     * 全局内存的原子布局。
     * shape为[行数(线程块线程数/每行线程数)，每行的线程数]
     * stride为[每行线程数，1]
     **/
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, /* 如果使用async copy，则使用SM80_CP_ASYNC_CACHEGLOBAL */
        DefaultCopy>;                               /* 否则使用默认的拷贝DefaulCopy */

    /**
     * 全局内存的 Q、K、V 和 O 矩阵的tile拷贝操作。
     * make_tiled_copy函数负责生成一个TiledCopy对象，每个tile的Layout为(1, 8)
     * GmemTiledCopyQKV实际上是一个TiledCopy对象
     */
    using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                                                      GmemLayoutAtom{},
                                                      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per read
    using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                                    GmemLayoutAtom{},
                                                    Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per store

    /**
     * 定义全局内存的原子输出累积布局
     */
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_16, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <_8, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
};

// Is_V_in_regs is an option to reduce smem usage, but will increase register pressue.
// No_double_buffer is another option to reduce smem usage, but will slow things down.
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
         bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;
    static constexpr bool No_double_buffer = No_double_buffer_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    using TiledMmaSdP = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        Tile<Int<16 * AtomLayoutMSdP>, Int<16 * kNWarps / AtomLayoutMSdP>, _16>>;

    using TiledMmadKV = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
        Tile<Int<16 * AtomLayoutNdKV>, Int<16 * kNWarps / AtomLayoutNdKV>, _16>>;

    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        Tile<Int<16 * AtomLayoutMdQ>, Int<16 * kNWarps / AtomLayoutMdQ>, _16>>;

    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(
        SmemLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        // SmemLayoutAtomQdO{},
        SmemLayoutAtomKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutKtransposed{}));

    // TODO: generalize to other values of kBlockN
    // TODO: what should be the Swizzle here? 3 is faster than 1, and 1 is faster than 2
    // static constexpr int kPBlockN = kBlockN;
    // Temporarily disabling this for hdim 256 on sm86 and sm89
    // static_assert(kBlockN >= 64);
    static_assert(kBlockN >= 32);
    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
    // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2 : 3);
    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                           Stride<Int<kPBlockN>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{}, make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutQdOtransposed = decltype(
        composition(SmemLayoutQdO{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    using SmemLayoutAtomdKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(
        SmemLayoutAtomdKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    // Double buffer for sQ
    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * (No_double_buffer ? 2 : 3) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemdSSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
    static constexpr int kSmemSize = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)
           : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)));
    static constexpr int kSmemSize1colblock = kSmemQdOSize
        + (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + kSmemPSize
           : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize));

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using GmemTiledCopydQaccumAtomicAdd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        Layout<Shape <_8, _32>,  // Thread layout, 8 threads per row
                               Stride<_32, _1>>{},
                        Layout<Shape < _1, _1>>{}));  // Val layout, 1 val per store

};

////////////////////////////////////////////////////////////////////////////////////////////////////
