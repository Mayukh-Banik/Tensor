#include "../Tensor/core/Tensor.cuh"

#include <gtest/gtest.h>

template <typename T>
bool arePropsValid(Tensor<T> &t, uint64_t len, uint64_t ndim, uint64_t *shape,
				   uint64_t *strides, uint64_t elementCount, bool Alloc = true)
{
	if (Alloc == true)
	{
		if (t.buf == NULL)
		{
			return false;
		}
	}
	if (t.len != len)
	{
		return false;
	}
	if (t.ndim != ndim)
	{
		return false;
	}
	for (uint64_t i = 0; i < ndim; i++)
	{
		if (t.strides[i] != strides[i] || t.shape[i] != shape[i])
		{
			return false;
		}
	}
	if (t.elementCount != elementCount)
	{
		return false;
	}
	return true;
}

TEST(Defualt_Constructor, heapAllocation)
{
	Tensor<double> *x;
	EXPECT_NO_THROW(x = new Tensor<double>());
	EXPECT_NO_THROW(delete (x));
}

TEST(Defualt_Constructor, stackAllocation)
{
	EXPECT_NO_THROW(Tensor<double> x = Tensor<double>());
}

TEST(Defualt_Constructor, getZero)
{
	Tensor<double> x = Tensor<double>();
	EXPECT_ANY_THROW(x.getIndex(0));
}

TEST(Defualt_Constructor, setZero)
{
	Tensor<double> x = Tensor<double>();
	EXPECT_ANY_THROW(x.setIndex(0, 0));
}

TEST(Defualt_Constructor, properties)
{
	Tensor<double> x = Tensor<double>();
	EXPECT_TRUE(arePropsValid(x, 0, 0, NULL, NULL, 0, false));
}

TEST(Scalar_Constructor, heapAllocation)
{
	Tensor<double> *x;
	EXPECT_NO_THROW(x = new Tensor<double>(1));
	EXPECT_NO_THROW(delete (x));
}

TEST(Scalar_Constructor, stackAllocation)
{
	EXPECT_NO_THROW(Tensor<double> x = Tensor<double>(2));
}