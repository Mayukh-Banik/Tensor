#include "../Tensor/core/Tensor.cuh"

#include <gtest/gtest.h>

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
	EXPECT_EQ(0, arePropsValid(x, 0, 0, NULL, NULL, 0, false));
	// EXPECT_TRUE(arePropsValid(x, 0, 0, NULL, NULL, 0, false));
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

TEST(Scalar_Constructor, inBoundSetting)
{
	double y = 2;
	Tensor<double> x = Tensor<double>(y);
	EXPECT_NO_THROW(x.getIndex(0));
	EXPECT_EQ(y, x.getIndex(0));
	EXPECT_ANY_THROW(x.getIndex(1));
	EXPECT_NO_THROW(x.setIndex(0, 1));
	EXPECT_EQ(3, x.setIndex(0, 3));
	EXPECT_ANY_THROW(x.setIndex(5, 5));
}

TEST(Scalar_Constructor, properties)
{
	Tensor<double> x = Tensor<double>(1);
	EXPECT_EQ(0, arePropsValid(x, sizeof(double), 0, NULL, NULL, 1));
}

TEST(ShapeNdim_Constructor, heapAllocation)
{
    uint64_t shape[] = {3, 4};
	Tensor<double>* x;
    EXPECT_NO_THROW(x = new Tensor<double>(shape, 2));
    EXPECT_NO_THROW(delete (x));
}

TEST(ShapeNdim_Constructor, stackAllocation)
{
    uint64_t shape[] = {2, 5};
    EXPECT_NO_THROW(Tensor<double> x = Tensor<double>(shape, 2));
}

TEST(ShapeNdim_Constructor, properties)
{
    uint64_t shape[] = {3, 4};
    uint64_t strides[] = {4 * sizeof(double), sizeof(double)}; // Row-major layout
    uint64_t ndim = 2;
    uint64_t elementCount = 3 * 4;
    uint64_t len = elementCount * sizeof(double);

    Tensor<double> x = Tensor<double>(shape, ndim);
    EXPECT_EQ(0, arePropsValid(x, len, ndim, shape, strides, elementCount));
}

TEST(ShapeNdim_Constructor, invalidAccess)
{
    uint64_t shape[] = {3, 3};
    Tensor<double> x = Tensor<double>(shape, 2);
    EXPECT_ANY_THROW(x.getIndex(9));
    EXPECT_NO_THROW(x.setIndex(0, 1.5));
    EXPECT_ANY_THROW(x.setIndex(10, 2.5));
}

TEST(Vector_Shape_Constructor, heapAllocation)
{
    std::vector<uint64_t> shape = {3, 4};
	Tensor<double> *x;
    EXPECT_NO_THROW(x = new Tensor<double>(shape));
    EXPECT_NO_THROW(delete (x));
}

TEST(Vector_Shape_Constructor, stackAllocation)
{
    std::vector<uint64_t> shape = {2, 5};
    EXPECT_NO_THROW(Tensor<double> x = Tensor<double>(shape));
}

TEST(Vector_Shape_Constructor, properties)
{
    std::vector<uint64_t> shape = {3, 4};
    uint64_t strides[] = {4 * sizeof(double), sizeof(double)}; // Row-major layout
    uint64_t ndim = shape.size();
    uint64_t elementCount = 3 * 4;
    uint64_t len = elementCount * sizeof(double);

    Tensor<double> x = Tensor<double>(shape);
    EXPECT_EQ(0, arePropsValid(x, len, ndim, shape.data(), strides, elementCount));
}

TEST(Vector_Shape_Constructor, invalidAccess)
{
    std::vector<uint64_t> shape = {3, 3};
    Tensor<double> x = Tensor<double>(shape);
    EXPECT_ANY_THROW(x.getIndex(9));
    EXPECT_NO_THROW(x.setIndex(0, 1.5));
    EXPECT_ANY_THROW(x.setIndex(10, 2.5));
}








