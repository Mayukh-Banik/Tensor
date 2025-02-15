#pragma once

#include "Tensor.cuh"

bool areShapesSame(const uint64_t* shape1, const uint64_t* shape2, const uint64_t ndim1, const uint64_t ndim2)
{
    if (ndim1 != ndim2)
    {
        return false;
    }
    for (uint64_t i = 0; i < ndim1; i++)
    {
        if (shape1[i] != shape2[i])
        {
            return false;
        }
    }
    return true;
}

template <typename A, typename B>
int operableLinearly(const Tensor<A>& a, const Tensor<B>& b)
{
    if (a.isScalar && b.isScalar)
    {
        return 0;
    }
    else if (a.isScalar && !b.isScalar)
    {
        return 1;
    }
    else if (!a.isScalar && b.isScalar)
    {
        return 2;
    }
    else if (!a.isScalar && !b.isScalar)
    {
        if (areShapesSame(a.shape, b.shape, a.ndim, b.ndim))
        {
            return 3;
        }
    }
    return -1;

}