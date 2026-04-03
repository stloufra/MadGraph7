#include "../kernels/kernels.h"
#include "device.h"
#include "tensor.h"

using namespace madspace;
using namespace madspace::gpu;
using namespace madspace::kernels;

std::pair<void*, Tensor> GpuDevice::allocate(std::size_t size) const {
    activate();
    void* ptr;
    check_error(gpuMalloc(&ptr, size));
    return {ptr, Tensor()};
}

void GpuDevice::free(void* ptr) const {
    activate();
    check_error(gpuFree(ptr));
}

void GpuDevice::memcpy(void* to, void* from, std::size_t size) const {
    activate();
    check_error(gpuMemcpy(to, from, size, gpuMemcpyDefault));
}

void GpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    activate();
    AsyncGpuDevice(*this, gpuStreamPerThread).tensor_copy(source, target);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_zero(Tensor& tensor) const {
    activate();
    AsyncGpuDevice(*this, gpuStreamPerThread).tensor_zero(tensor);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    activate();
    AsyncGpuDevice(*this, gpuStreamPerThread).tensor_add(source, target);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    activate();
    check_error(
        gpuMemcpy(target.data(), source.data(), source.byte_size(), gpuMemcpyDefault)
    );
}

MemPool::MemPool(const SizeVec pool_factors, const std::vector<AllocItem>& allocs) :
    _allocs(allocs) {
    _pools.reserve(pool_factors.size());
    for (auto& factor : pool_factors) {
        _pools.push_back({
            .size_factor = factor,
            .batch_size = 0,
            .parent_tensor = Tensor(),
        });
    }
}

std::pair<void*, Tensor>
MemPool::allocate(std::size_t size, const GpuDevice& device, gpuStream_t stream) {
    AllocItem& alloc = _allocs.at(_alloc_index);
    ++_alloc_index;
    PoolItem& pool = _pools.at(alloc.pool_index);
    if (size % alloc.size_factor != 0) {
        throw std::runtime_error("inconsistent pool allocation");
    }
    std::size_t batch_size = size / alloc.size_factor;
    if (!pool.parent_tensor) {
        pool.batch_size = batch_size;
        AsyncGpuDevice async_device(device, stream);
        pool.parent_tensor = Tensor(
            DataType::dt_float, {(batch_size * pool.size_factor + 7) / 8}, async_device
        );
    } else if (batch_size != pool.batch_size) {
        throw std::runtime_error("inconsistent pool allocation");
    }
    return {
        static_cast<uint8_t*>(pool.parent_tensor.data()) +
            pool.batch_size * alloc.offset,
        pool.parent_tensor
    };
}

std::pair<void*, Tensor> AsyncGpuDevice::allocate(std::size_t size) const {
    if (_mem_pool != nullptr) {
        return _mem_pool->allocate(size, _device, _stream);
    } else {
        void* ptr;
        check_error(gpuMallocAsync(&ptr, size, _stream));
        return {ptr, Tensor()};
    }
}

void AsyncGpuDevice::free(void* ptr) const { check_error(gpuFreeAsync(ptr, _stream)); }

void AsyncGpuDevice::memcpy(void* to, void* from, std::size_t size) const {
    check_error(gpuMemcpyAsync(to, from, size, gpuMemcpyDefault, _stream));
}

void AsyncGpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    if (source.dtype() == DataType::dt_float && target.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic<kernel_copy<GpuTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    } else if (source.dtype() == DataType::dt_int &&
               target.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic<kernel_copy_int<GpuTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    } else {
        throw std::runtime_error("invalid dtype in copy");
    }
}

void AsyncGpuDevice::tensor_zero(Tensor& tensor) const {
    if (tensor.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic<kernel_zero<GpuTypes>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), *this
        );
    } else if (tensor.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic<kernel_zero_int<GpuTypes>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), *this
        );
    } else {
        throw std::runtime_error("invalid dtype in zero");
    }
}

void AsyncGpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_foreach_dynamic<kernel_add_inplace<GpuTypes>, 1, 1>(
        {&source}, {&target}, target.size(0), *this
    );
}

void AsyncGpuDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    check_error(gpuMemcpyAsync(
        target.data(), source.data(), source.byte_size(), gpuMemcpyDefault, _stream
    ));
}

extern "C" int device_count() {
    int device_count;
    check_error(gpuGetDeviceCount(&device_count));
    return device_count;
}

extern "C" DevicePtr get_device(int index) { return &GpuDevice::instance(index); }
