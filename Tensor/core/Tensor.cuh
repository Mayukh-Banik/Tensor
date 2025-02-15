#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <functional>
#include <iomanip>

/**
 * An example Templated class thats recreating NumPy on the GPU, with python bindings for float only (TODO).
 */
template <typename T>
class Tensor
{
private:
	/**
	 * Sets all elements of cudaProperties. Only works on a single GPU currently.
	 */
	void setDeviceProperties();

public:
	// Python buffer protocol section"
	T *buf = NULL;
	// length in bytes
	uint64_t len;
	const bool readOnly = false;
	const uint64_t itemsize = sizeof(T);
	const std::string format = "d";
	uint64_t ndim;
	// Allocated with malloc
	uint64_t *shape = NULL;
	// Allocated with malloc
	uint64_t *strides = NULL;
	// End Python buffer protocol section

	// Current CUDA device number
	int deviceNumber;
	// Number of numbers in Tensor
	uint64_t elementCount;

	bool isScalar = false;

	/**
	 * Properties of current CUDA device
	 */
	struct
	{
		char name[256];
		uint32_t warpSize;
		uint32_t maxThreadsPerBlock;
		uint32_t maxThreadsDim[3];
		uint32_t maxGridSize[3];
		uint32_t maxGrids;
	} cudaProperties;

	/**
	 * Creates an empty Tensor. every value is set to 0.
	 */
	Tensor();

	/**
	 * Creates a scalar Tensor.
	 * @param value Data type and scalar value of Tensor that will be created
	 */
	Tensor(T value);

	/**
	 * Creates a Tensor with dimensions. Uninitialized
	 * @param shape array of shape to create
	 * @param ndim how much of shape to look at. [0, ndim]
	 */
	Tensor(const uint64_t *shape, const uint64_t ndim);
	
	/**
	 * Shape from vector
	 * @todo Fix ambiguity with values
	 */
	Tensor(const std::vector<uint64_t> &shape);
	/**
	 *
	 */
	Tensor(const T *values, const uint64_t *shape, const uint64_t ndim);
	Tensor(const T *values, const std::vector<uint64_t> &shape);
	~Tensor() noexcept;

	std::string toString(bool debug = false) const;

	T getIndex(const uint64_t index);
	T setIndex(const uint64_t index, T value);
};

template <typename T>
void Tensor<T>::setDeviceProperties()
{
	cudaDeviceProp prop;
	cudaError_t err = cudaGetDeviceCount(&this->deviceNumber);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
	if (this->deviceNumber == 0)
	{
		throw std::runtime_error("No CUDA GPUS found");
	}
	err = cudaGetDevice(&this->deviceNumber);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
	err = cudaGetDeviceProperties(&prop, this->deviceNumber);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
	strncpy(this->cudaProperties.name, prop.name, sizeof(this->cudaProperties.name) - 1);
	this->cudaProperties.name[sizeof(this->cudaProperties.name) - 1] = '\0';
	this->cudaProperties.warpSize = prop.warpSize;
	this->cudaProperties.maxThreadsPerBlock = prop.maxThreadsPerBlock;
	this->cudaProperties.maxThreadsDim[0] = prop.maxThreadsDim[0];
	this->cudaProperties.maxThreadsDim[1] = prop.maxThreadsDim[1];
	this->cudaProperties.maxThreadsDim[2] = prop.maxThreadsDim[2];
	this->cudaProperties.maxGridSize[0] = prop.maxGridSize[0];
	this->cudaProperties.maxGridSize[1] = prop.maxGridSize[1];
	this->cudaProperties.maxGridSize[2] = prop.maxGridSize[2];
	this->cudaProperties.maxGrids = prop.maxGridSize[0];
}

template <typename T>
Tensor<T>::Tensor()
{
	setDeviceProperties();
	this->elementCount = 0;
	this->len = 0;
	this->ndim = 0;
}

template <typename T>
Tensor<T>::Tensor(T value)
{
	setDeviceProperties();
	this->elementCount = 1;
	this->len = this->itemsize;
	this->ndim = 0;
	this->shape = NULL;
	this->strides = NULL;
	cudaError_t err = cudaMalloc((void **)&this->buf, this->len);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
	err = cudaMemcpy(this->buf, &value, this->len, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
	this->isScalar = true;
}

template <typename T>
Tensor<T>::Tensor(const uint64_t *shape, const uint64_t ndim)
{
	if (shape == NULL || shape == nullptr || ndim == 0)
	{
		throw std::runtime_error("Shape cannot be NULL, nullptr, or ndim == 0.");
	}
	setDeviceProperties();
	this->shape = (uint64_t *)malloc(sizeof(*shape) * ndim);
	this->strides = (uint64_t *)malloc(sizeof(*shape) * ndim);
	this->elementCount = 1;
	this->ndim = ndim;
	for (uint64_t i = 0; i < ndim; i++)
	{
		this->elementCount = this->elementCount * shape[i];
		this->shape[i] = shape[i];
	}
	this->strides[ndim - 1] = this->itemsize;
	for (int64_t i = ndim - 2; i >= 0; i--)
	{
		this->strides[i] = this->strides[i + 1] * this->shape[i + 1];
	}
	this->len = this->elementCount * this->itemsize;
	cudaError_t err = cudaMalloc((void **)&this->buf, this->len);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

template <typename T>
Tensor<T>::Tensor(const std::vector<uint64_t> &shape)
{
	if (shape.empty())
	{
		throw std::runtime_error("Shape vector cannot be empty.");
	}
	setDeviceProperties();
	this->ndim = shape.size();
	this->shape = (uint64_t *)malloc(sizeof(uint64_t) * this->ndim);
	this->strides = (uint64_t *)malloc(sizeof(uint64_t) * this->ndim);
	this->elementCount = 1;
	for (uint64_t i = 0; i < this->ndim; i++)
	{
		this->shape[i] = shape[i];
		this->elementCount *= shape[i];
	}
	this->strides[ndim - 1] = this->itemsize;
	for (int64_t i = ndim - 2; i >= 0; i--)
	{
		this->strides[i] = this->strides[i + 1] * this->shape[i + 1];
	}
	this->len = this->elementCount * this->itemsize;
	cudaError_t err = cudaMalloc((void **)&this->buf, this->len);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

template <typename T>
Tensor<T>::Tensor(const T *values, const uint64_t *shape, const uint64_t ndim) : Tensor<T>::Tensor(shape, ndim)
{
	cudaError_t err = cudaMemcpy(this->buf, values, this->len, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

template <typename T>
Tensor<T>::Tensor(const T *values, const std::vector<uint64_t> &shape) : Tensor<T>::Tensor(shape)
{
	cudaError_t err = cudaMemcpy(this->buf, values, this->len, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

template <typename T>
Tensor<T>::~Tensor() noexcept
{
	if (this->shape != NULL && this->shape != nullptr)
	{
		free(this->shape);
	}
	if (this->strides != NULL && this->strides != nullptr)
	{
		free(this->strides);
	}
	if (this->buf != NULL && this->buf != nullptr)
	{
		cudaError_t err = cudaFree(this->buf);
		if (err != cudaSuccess)
		{
			std::cerr << cudaGetErrorString(err) << std::endl;
		}
	}
}

template <typename T>
std::string Tensor<T>::toString(bool Debug) const
{
	std::ostringstream s;
	if (Debug)
	{
		s << "Data location: " << this->buf << std::endl;
		s << "Length in Bytes: " << this->len << std::endl;
		s << "Element Count: " << this->elementCount << std::endl;
		s << "Number of Dimensions: " << this->ndim << std::endl;
		s << "Shape: (";
		for (uint64_t i = 0; i < this->ndim; i++)
		{
			s << this->shape[i];
			if (i < this->ndim - 1)
				s << ", ";
		}
		s << ")" << std::endl;
		s << "Strides: (";
		for (uint64_t i = 0; i < this->ndim; i++)
		{
			s << this->strides[i];
			if (i < this->ndim - 1)
				s << ", ";
		}
		s << ")" << std::endl;
		s << "Device: cuda:" << this->deviceNumber << std::endl;
	}
	if (this->len == 0)
	{
		s << "[]" << std::endl;
		return s.str();
	}
	T *data = (T *)malloc(this->len);
	cudaMemcpy(data, this->buf, this->len, cudaMemcpyDeviceToHost);
	std::function<void(int, uint64_t)> printArray = [&](uint64_t dim, uint64_t offset)
	{
		if (dim < this->ndim - 1)
		{
			s << "[";
			for (uint64_t i = 0; i < shape[dim]; i++)
			{
				printArray(dim + 1, offset + i * strides[dim]);
				if (i < shape[dim] - 1)
				{
					s << std::endl;
					for (uint64_t j = 0; j <= dim; j++)
						s << " ";
				}
			}
			s << "]";
		}
		else
		{
			s << "[";
			for (uint64_t i = 0; i < shape[dim]; i++)
			{
				T value = data[offset / sizeof(T) + i * strides[dim] / sizeof(T)];
				s << std::fixed << std::setprecision(6) << value;
				if (i < shape[dim] - 1)
					s << ", ";
			}
			s << "]";
		}
	};
	if (this->ndim == 0)
	{
		s << "[" << std::fixed << std::setprecision(6) << data[0] << "]";
	}
	else
	{
		printArray(0, 0);
	}
	s << std::endl;
	free(data);
	return s.str();
}

template <typename T>
T Tensor<T>::getIndex(const uint64_t index)
{
	if (index > this->elementCount || this->elementCount == 0)
	{
		throw std::invalid_argument("Index greater than number of elements");
	}
	T val;
	cudaError_t err = cudaMemcpy(&val, this->buf + index, this->itemsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
	return val;
}

template <typename T>
T Tensor<T>::setIndex(const uint64_t index, T value)
{
	if (index > this->elementCount)
	{
		throw std::invalid_argument("Index greater than number of elements");
	}
	cudaError_t err = cudaMemcpy(this->buf + index, &value, this->itemsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
	return value;
}

































