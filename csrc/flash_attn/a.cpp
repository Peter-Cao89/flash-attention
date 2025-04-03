
#include <cutlass/numeric_types.h>

void f()
{

    [&]
    {
        if (!params.is_bf16)
        {
            using elem_type = cutlass::half_t;
            return [&]
            {
                [&]
                {
                    if (params.d <= 32)
                    {
                        constexpr static int kHeadDim = 32;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 64)
                    {
                        constexpr static int kHeadDim = 64;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    { if (params.num_splits <= 1 && !force_split_kernel) { run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream); } else { run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream); } }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 96)
                    {
                        constexpr static int kHeadDim = 96;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 128)
                    {
                        constexpr static int kHeadDim = 128;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 160)
                    {
                        constexpr static int kHeadDim = 160;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 192)
                    {
                        constexpr static int kHeadDim = 192;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 224)
                    {
                        constexpr static int kHeadDim = 224;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 256)
                    {
                        constexpr static int kHeadDim = 256;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                }();
            }();
        }
        else
        {
            using elem_type = cutlass::bfloat16_t;
            return [&]
            {
                [&]
                {
                    if (params.d <= 32)
                    {
                        constexpr static int kHeadDim = 32;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 64)
                    {
                        constexpr static int kHeadDim = 64;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 96)
                    {
                        constexpr static int kHeadDim = 96;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 128)
                    {
                        constexpr static int kHeadDim = 128;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 160)
                    {
                        constexpr static int kHeadDim = 160;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 192)
                    {
                        constexpr static int kHeadDim = 192;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 224)
                    {
                        constexpr static int kHeadDim = 224;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                    else if (params.d <= 256)
                    {
                        constexpr static int kHeadDim = 256;
                        return [&]
                        {
                            [&]
                            {
                                if (params.is_causal)
                                {
                                    constexpr static bool Is_causal = true;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }

                                        
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                                else
                                {
                                    constexpr static bool Is_causal = false;
                                    return [&]
                                    {
                                        if (params.num_splits <= 1 && !force_split_kernel)
                                        {
                                            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                        else
                                        {
                                            run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                                        }
                                    }();
                                }
                            }();
                        }();
                    }
                }();
            }();
        }
    }()
}