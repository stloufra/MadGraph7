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

MemPool::MemPool(
    const GpuDevice& device,
    const std::vector<std::pair<std::size_t, std::size_t>>& cached_sizes
) :
    _device(device) {
    std::size_t pool_count = 0;
    for (auto& [pool_index, size] : cached_sizes) {
        if (pool_index >= pool_count) {
            pool_count = pool_index + 1;
        }
    }
    _pools.resize(pool_count);

    for (auto& [pool_index, size] : cached_sizes) {
        auto& pool = _pools.at(pool_index);
        std::size_t word_count = (size + 7) / 8;
        pool.parent_tensor = Tensor(DataType::dt_float, {word_count}, device);
        pool.size = word_count * 8;
        pool.needed_size = word_count * 8;
    }
}

MemPool::~MemPool() {
    for (PoolItem& pool : _pools) {
        for (auto& [size, item] : pool.free_pointers) {
            auto& [ptr, parent] = item;
            if (!parent) {
                check_error(gpuFree(ptr));
            }
        }
    }
}

std::pair<void*, Tensor> MemPool::allocate(std::size_t pool_index, std::size_t size) {
    if (pool_index >= _pools.size()) {
        _pools.resize(pool_index);
    }
    PoolItem& pool = _pools.at(pool_index);
    if (auto search = pool.free_pointers.find(size);
        search != pool.free_pointers.end()) {
        std::pair<void*, Tensor> ret = *search->second;
        pool.free_pointers.erase(search);
        return ret;
    } else if (pool.capacity - pool.size >= size) {
        void* ptr = &static_cast<uint8_t*>(pool.parent_tensor.data())[pool.size];
        pool.size = (pool.size + size + 7) / 8 * 8;
        _allocs[ptr] = {
            .pool_index = pool_index,
            .size = size,
            .parent_tensor = pool.parent_tensor,
        };
        return {ptr, pool.parent_tensor};
    } else {
        void* ptr;
        check_error(gpuMalloc(&ptr, size));
        _allocs[ptr] = {
            .pool_index = pool_index,
            .size = size,
            .parent_tensor = Tensor(),
        };
        pool.needed_size += (size + 7) / 8 * 8;
        return {ptr, Tensor()};
    }
}

void MemPool::free(void* ptr) {
    auto search = _allocs.find(ptr) if (search == _allocs.end()) {
        throw std::runtime_error("address was not allocated using this pool");
    }
    auto& alloc = search->second;
    _pools.at(alloc.pool_index)
        .free_pointers.emplace(alloc.size, {ptr, alloc.parent_tensor});
    _allocs.erase(search);
}

std::vector<std::pair<std::size_t, std::size_t>> MemPool::total_sizes() const {
    std::vector<std::pair<std::size_t, std::size_t>> ret;
    ret.reserve(_pools.size());
    for (std::size_t index = 0; PoolItem& pool : _pools) {
        if (pool.needed_size > 0) {
            ret.push_back({index, pool.needed_size});
        }
        ++index;
    }
    return ret;
}

std::pair<void*, Tensor>
AsyncGpuDevice::allocate(std::size_t size, AllocHint hint) const {
    if (_mem_pool != nullptr && hint != AllocHint::normal) {
        return _mem_pool->allocate(static_cast<std::size_t>(hint) - 1, size);
    } else {
        _device.allocate(size, hint);
        // void* ptr;
        // check_error(gpuMallocAsync(&ptr, size, _stream));
        // return {ptr, Tensor()};
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
