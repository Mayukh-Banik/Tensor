#pragma once

#include "../core/Tensor.cuh"

namespace tensor
{
    template <typename A, typename B, typename C>
    Tensor<A>* add(const Tensor<B>& a, const Tensor<C>& b);

    template <typename A, typename B, typename C>
    Tensor<A>* add(const Tensor<B>& a, const C b);
}


template <typename A, typename B, typename C>
Tensor<A>* tensor::add(const Tensor<B>& a, const C b) 
{
    return add(a, Tensor<C>(b));
}

template <typename A, typename B, typename C>
Tensor<A>* tensor::add(const Tensor<B>& a, const Tensor<C>& b) 
{
    return nullptr;
}