#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<int N> struct MemRefDescriptor
{
    float*  allocated;
    float*  aligned;
    int64_t offset;
    int64_t sizes[N];
    int64_t strides[N];
};

float* load_binary_file(const char* path, int64_t num_elements)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s\n", path);
        exit(1);
    }

    int ndim = 0;
    if (fread(&ndim, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error: Failed to read header (ndim) from %s\n", path);
        fclose(f);
        exit(1);
    }

    if (fseek(f, ndim * sizeof(int), SEEK_CUR) != 0) {
        fprintf(stderr, "Error: Failed to skip shape header in %s\n", path);
        fclose(f);
        exit(1);
    }

    long expectedSize = num_elements * sizeof(float);

    float* data = (float*)aligned_alloc(64, expectedSize);
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for %s\n", path);
        fclose(f);
        exit(1);
    }

    size_t read_count = fread(data, sizeof(float), num_elements, f);
    if (read_count != num_elements) {
        fprintf(stderr,
                "Warning: Read %zu elements, expected %ld from %s\n",
                read_count,
                num_elements,
                path);
    }

    fclose(f);
    return data;
}


template<int N> MemRefDescriptor<N> create_memref(float* data, const std::vector<int64_t>& shape)
{
    MemRefDescriptor<N> desc;
    desc.allocated = data;
    desc.aligned   = data;
    desc.offset    = 0;

    int64_t stride = 1;
    for (int i = N - 1; i >= 0; --i) {
        desc.sizes[i]   = shape[i];
        desc.strides[i] = stride;
        stride *= shape[i];
    }
    return desc;
}

extern "C" {

MemRefDescriptor<1> cherry_read_weight_1d_768_f32(char* p_alloc, char* p_align, int64_t p_off,
                                                  int64_t p_size, int64_t p_stride, int64_t dim0)
{
    printf("Loading 1D weight: %s [%ld]\n", p_align, dim0);
    float* data = load_binary_file(p_align, dim0);
    return create_memref<1>(data, {dim0});
}

MemRefDescriptor<2> cherry_read_weight_2d_12_768_f32(char* p_alloc, char* p_align, int64_t p_off,
                                                     int64_t p_size, int64_t p_stride, int64_t dim0,
                                                     int64_t dim1)
{
    printf("Loading 2D weight: %s [%ld, %ld]\n", p_align, dim0, dim1);
    float* data = load_binary_file(p_align, dim0 * dim1);
    return create_memref<2>(data, {dim0, dim1});
}

MemRefDescriptor<2> cherry_read_weight_2d_32000_768_f32(char* p_alloc, char* p_align, int64_t p_off,
                                                        int64_t p_size, int64_t p_stride,
                                                        int64_t dim0, int64_t dim1)
{
    printf("Loading 2D weight: %s [%ld, %ld]\n", p_align, dim0, dim1);
    float* data = load_binary_file(p_align, dim0 * dim1);
    return create_memref<2>(data, {dim0, dim1});
}

MemRefDescriptor<2> cherry_read_weight_2d_768_32000_f32(char* p_alloc, char* p_align, int64_t p_off,
                                                        int64_t p_size, int64_t p_stride,
                                                        int64_t dim0, int64_t dim1)
{
    printf("Loading 2D weight: %s [%ld, %ld]\n", p_align, dim0, dim1);
    float* data = load_binary_file(p_align, dim0 * dim1);
    return create_memref<2>(data, {dim0, dim1});
}

MemRefDescriptor<3> cherry_read_weight_3d_12_768_768_f32(char* p_alloc, char* p_align,
                                                         int64_t p_off, int64_t p_size,
                                                         int64_t p_stride, int64_t dim0,
                                                         int64_t dim1, int64_t dim2)
{
    printf("Loading 3D weight: %s [%ld, %ld, %ld]\n", p_align, dim0, dim1, dim2);
    float* data = load_binary_file(p_align, dim0 * dim1 * dim2);
    return create_memref<3>(data, {dim0, dim1, dim2});
}

MemRefDescriptor<3> cherry_read_weight_3d_12_768_2048_f32(char* p_alloc, char* p_align,
                                                          int64_t p_off, int64_t p_size,
                                                          int64_t p_stride, int64_t dim0,
                                                          int64_t dim1, int64_t dim2)
{
    printf("Loading 3D weight: %s [%ld, %ld, %ld]\n", p_align, dim0, dim1, dim2);
    float* data = load_binary_file(p_align, dim0 * dim1 * dim2);
    return create_memref<3>(data, {dim0, dim1, dim2});
}

MemRefDescriptor<3> cherry_read_weight_3d_12_2048_768_f32(char* p_alloc, char* p_align,
                                                          int64_t p_off, int64_t p_size,
                                                          int64_t p_stride, int64_t dim0,
                                                          int64_t dim1, int64_t dim2)
{
    printf("Loading 3D weight: %s [%ld, %ld, %ld]\n", p_align, dim0, dim1, dim2);
    float* data = load_binary_file(p_align, dim0 * dim1 * dim2);
    return create_memref<3>(data, {dim0, dim1, dim2});
}

}   // extern "C"
