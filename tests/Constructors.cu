#include "../Tensor/core/Tensor.cuh"

#include <gtest/gtest.h>

#define DATA_TYPE double

template <typename T>
int arePropsValid(Tensor<T> &t, uint64_t len, uint64_t ndim, uint64_t *shape,
				  uint64_t *strides, uint64_t elementCount, bool Alloc = true)
{
	if (Alloc == true)
	{
		if (t.buf == NULL)
		{
			return 1;
		}
	}
	if (t.len != len)
	{
		return 2;
	}
	if (t.ndim != ndim)
	{
		return 3;
	}
	for (uint64_t i = 0; i < ndim; i++)
	{
		if (t.strides[i] != strides[i] || t.shape[i] != shape[i])
		{
			return 4;
		}
	}
	if (t.elementCount != elementCount)
	{
		return 5;
	}
	return 0;
}

TEST(Defualt_Constructor, heapAllocation)
{
	Tensor<DATA_TYPE> *x;
	EXPECT_NO_THROW(x = new Tensor<DATA_TYPE>());
	EXPECT_NO_THROW(delete (x));
}

TEST(Defualt_Constructor, stackAllocation)
{
	EXPECT_NO_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>());
}

TEST(Defualt_Constructor, getZero)
{
	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>();
	EXPECT_ANY_THROW(x.getIndex(0));
}

TEST(Defualt_Constructor, setZero)
{
	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>();
	EXPECT_ANY_THROW(x.setIndex(0, 0));
}

TEST(Defualt_Constructor, properties)
{
	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>();
	EXPECT_EQ(0, arePropsValid(x, 0, 0, NULL, NULL, 0, false));
}

TEST(Scalar_Constructor, heapAllocation)
{
	Tensor<DATA_TYPE> *x;
	EXPECT_NO_THROW(x = new Tensor<DATA_TYPE>(1));
	EXPECT_NO_THROW(delete (x));
}

TEST(Scalar_Constructor, stackAllocation)
{
	EXPECT_NO_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(2));
}

TEST(Scalar_Constructor, inBoundSetting)
{
	DATA_TYPE y = 2;
	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(y);
	EXPECT_NO_THROW(x.getIndex(0));
	EXPECT_EQ(y, x.getIndex(0));
	EXPECT_ANY_THROW(x.getIndex(1));
	EXPECT_NO_THROW(x.setIndex(0, 1));
	EXPECT_EQ(3, x.setIndex(0, 3));
	EXPECT_ANY_THROW(x.setIndex(5, 5));
}

TEST(Scalar_Constructor, properties)
{
	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(1);
	EXPECT_EQ(0, arePropsValid(x, sizeof(DATA_TYPE), 0, NULL, NULL, 1));
}

TEST(ShapeNdim_Constructor, heapAllocation)
{
	uint64_t shape[] = {3, 4};
	Tensor<DATA_TYPE> *x;
	EXPECT_NO_THROW(x = new Tensor<DATA_TYPE>(shape, 2));
	EXPECT_NO_THROW(delete (x));
}

TEST(ShapeNdim_Constructor, stackAllocation)
{
	uint64_t shape[] = {2, 5};
	EXPECT_NO_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(shape, 2));
}

TEST(ShapeNdim_Constructor, properties)
{
	uint64_t shape[] = {3, 4};
	uint64_t strides[] = {4 * sizeof(DATA_TYPE), sizeof(DATA_TYPE)};
	uint64_t ndim = 2;
	uint64_t elementCount = 3 * 4;
	uint64_t len = elementCount * sizeof(DATA_TYPE);

	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(shape, ndim);
	EXPECT_EQ(0, arePropsValid(x, len, ndim, shape, strides, elementCount));
}

TEST(ShapeNdim_Constructor, invalidAccess)
{
	uint64_t shape[] = {3, 3};
	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(shape, 2);
	EXPECT_ANY_THROW(x.getIndex(9));
	EXPECT_NO_THROW(x.setIndex(0, 1.5));
	EXPECT_ANY_THROW(x.setIndex(10, 2.5));
}

TEST(Vector_Shape_Constructor, heapAllocation)
{
	std::vector<uint64_t> shape = {3, 4};
	Tensor<DATA_TYPE> *x;
	EXPECT_NO_THROW(x = new Tensor<DATA_TYPE>(shape));
	EXPECT_NO_THROW(delete (x));
}

TEST(Vector_Shape_Constructor, stackAllocation)
{
	std::vector<uint64_t> shape = {2, 5};
	EXPECT_NO_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(shape));
}

TEST(Vector_Shape_Constructor, properties)
{
	std::vector<uint64_t> shape = {3, 4};
	uint64_t strides[] = {4 * sizeof(DATA_TYPE), sizeof(DATA_TYPE)};
	uint64_t ndim = shape.size();
	uint64_t elementCount = 3 * 4;
	uint64_t len = elementCount * sizeof(DATA_TYPE);

	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(shape);
	EXPECT_EQ(0, arePropsValid(x, len, ndim, shape.data(), strides, elementCount));
}

TEST(Vector_Shape_Constructor, invalidAccess)
{
	std::vector<uint64_t> shape = {3, 3};
	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(shape);
	EXPECT_ANY_THROW(x.getIndex(9));
	EXPECT_NO_THROW(x.setIndex(0, 1.5));
	EXPECT_ANY_THROW(x.setIndex(10, 2.5));
}

TEST(Raw_Pointer_Shape_Array_Constructor, heapAllocation)
{
	DATA_TYPE values[] = {1, 2, 3, 4};
	uint64_t shape[] = {2, 2};
	Tensor<DATA_TYPE> *x;
	EXPECT_NO_THROW(x = new Tensor<DATA_TYPE>(values, shape, 2));
	EXPECT_NO_THROW(delete x);
}

TEST(Raw_Pointer_Shape_Array_Constructor, stackAllocation)
{
	DATA_TYPE values[] = {1, 2, 3, 4};
	uint64_t shape[] = {2, 2};
	EXPECT_NO_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(values, shape, 2));
}

TEST(Raw_Pointer_Shape_Array_Constructor, properties)
{
	DATA_TYPE values[] = {1, 2, 3, 4};
	uint64_t shape[] = {2, 2};
	uint64_t strides[] = {2 * sizeof(DATA_TYPE), sizeof(DATA_TYPE)};
	uint64_t ndim = 2;
	uint64_t elementCount = 4;
	uint64_t len = elementCount * sizeof(DATA_TYPE);

	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(values, shape, ndim);
	EXPECT_EQ(0, arePropsValid(x, len, ndim, shape, strides, elementCount));

	for (uint64_t i = 0; i < elementCount; i++)
	{
		EXPECT_EQ(values[i], x.getIndex(i));
	}
}

TEST(Raw_Pointer_Vector_Shape_Constructor, heapAllocation)
{
	DATA_TYPE values[] = {1, 2, 3, 4};
	std::vector<uint64_t> shape = {2, 2};
	Tensor<DATA_TYPE> *x;
	EXPECT_NO_THROW(x = new Tensor<DATA_TYPE>(values, shape));
	EXPECT_NO_THROW(delete x);
}

TEST(Raw_Pointer_Vector_Shape_Constructor, stackAllocation)
{
	DATA_TYPE values[] = {1, 2, 3, 4};
	std::vector<uint64_t> shape = {2, 2};
	EXPECT_NO_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(values, shape));
}

TEST(Cuda_Raw_Pointer_Shape_Array_Constructor, heapAllocation)
{
	const DATA_TYPE values[] = {1.0, 2.0, 3.0, 4.0};
	const uint64_t shape[] = {2, 2};
	Tensor<DATA_TYPE> *x;
	EXPECT_NO_THROW(x = new Tensor<DATA_TYPE>(values, shape, 2));
	EXPECT_NO_THROW(delete x);
}

TEST(Cuda_Raw_Pointer_Shape_Array_Constructor, stackAllocation)
{
	const DATA_TYPE values[] = {1.0, 2.0, 3.0, 4.0};
	const uint64_t shape[] = {2, 2};
	EXPECT_NO_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(values, shape, 2));
}

TEST(Cuda_Raw_Pointer_Shape_Array_Constructor, properties)
{
	const DATA_TYPE values[] = {1.0, 2.0, 3.0, 4.0};
	uint64_t shape[] = {2, 2};
	uint64_t strides[] = {2 * sizeof(DATA_TYPE), sizeof(DATA_TYPE)};
	uint64_t ndim = 2;
	uint64_t elementCount = 4;
	uint64_t len = elementCount * sizeof(DATA_TYPE);

	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(values, shape, ndim);
	EXPECT_EQ(0, arePropsValid(x, len, ndim, shape, strides, elementCount));

	DATA_TYPE hostBuffer[4];
	cudaError_t err = cudaMemcpy(hostBuffer, x.buf, len, cudaMemcpyDeviceToHost);
	EXPECT_EQ(cudaSuccess, err);
	for (uint64_t i = 0; i < elementCount; i++)
	{
		EXPECT_EQ(values[i], hostBuffer[i]);
	}
}

TEST(Cuda_Raw_Pointer_Vector_Shape_Constructor, heapAllocation)
{
	const DATA_TYPE values[] = {1.0, 2.0, 3.0, 4.0};
	const std::vector<uint64_t> shape = {2, 2};
	Tensor<DATA_TYPE> *x;
	EXPECT_NO_THROW(x = new Tensor<DATA_TYPE>(values, shape));
	EXPECT_NO_THROW(delete x);
}

TEST(Cuda_Raw_Pointer_Vector_Shape_Constructor, stackAllocation)
{
	const DATA_TYPE values[] = {1.0, 2.0, 3.0, 4.0};
	const std::vector<uint64_t> shape = {2, 2};
	EXPECT_NO_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(values, shape));
}

TEST(Cuda_Raw_Pointer_Vector_Shape_Constructor, properties)
{
	const DATA_TYPE values[] = {1.0, 2.0, 3.0, 4.0};
	std::vector<uint64_t> shape = {2, 2};
	uint64_t strides[] = {2 * sizeof(DATA_TYPE), sizeof(DATA_TYPE)};
	uint64_t ndim = shape.size();
	uint64_t elementCount = 4;
	uint64_t len = elementCount * sizeof(DATA_TYPE);

	Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(values, shape);
	EXPECT_EQ(0, arePropsValid(x, len, ndim, shape.data(), strides, elementCount));

	DATA_TYPE hostBuffer[4];
	cudaError_t err = cudaMemcpy(hostBuffer, x.buf, len, cudaMemcpyDeviceToHost);
	EXPECT_EQ(cudaSuccess, err);
	for (uint64_t i = 0; i < elementCount; i++)
	{
		EXPECT_EQ(values[i], hostBuffer[i]);
	}
}

TEST(Cuda_Raw_Pointer_Vector_Shape_Constructor, emptyShape)
{
	const DATA_TYPE values[] = {1.0};
	const std::vector<uint64_t> emptyShape = {};
	EXPECT_ANY_THROW(Tensor<DATA_TYPE> x = Tensor<DATA_TYPE>(values, emptyShape));
}

