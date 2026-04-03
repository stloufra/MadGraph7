#pragma once

#include "gpu_abstraction.h"
#include "madspace/madcode/function.h"
#include "madspace/runtime/runtime_base.h"
#include "madspace/runtime/tensor.h"

#include <memory>

namespace madspace {
namespace gpu {

class GpuRuntime : public Runtime {
public:
    struct Instruction {
        int opcode;
        SizeVec input_indices;
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        std::size_t batch_size_index;
        GpuRuntime& runtime;
        bool differentiable;
        std::size_t stream;
        SizeVec wait_events;
        int record_event;
        SizeVec backward_wait_events;
        int backward_record_event;
    };

    GpuRuntime(const Function& function, ContextPtr context);
    TensorVec run(const TensorVec& inputs) const override;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) const override;
    std::tuple<TensorVec, std::vector<std::tuple<std::string, Tensor>>> run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad
    ) const override;
    Context& context() { return *_context; }
    gpublasHandle_t gpublas_handle() { return _gpublas_handle.get(); }
    gpurandGenerator_t gpurand_generator() { return _gpurand_generator.get(); }

private:
    void

        std::vector<Instruction>
            _instructions;
    SizeVec _output_indices;
    std::size_t _input_count;
    TensorVec _locals_init;
    std::vector<bool> _requires_grad_init;
    std::vector<std::tuple<std::string, std::size_t>> _grad_global_indices;
    ContextPtr _context;
    ThreadResource<std::vector<gpuStream_t>> _streams;
    ThreadResource<std::vector<gpuEvent_t>> _events;
    std::vector<std::size_t> _wait_events;
    std::vector<std::size_t> _backward_wait_events;
    ThreadResource<gpublasHandle_t> _gpublas_handle;
    ThreadResource<gpurandGenerator_t> _gpurand_generator;
};

extern "C" Runtime*
build_runtime(const Function& function, ContextPtr context, bool concurrent);

} // namespace gpu
} // namespace madspace
