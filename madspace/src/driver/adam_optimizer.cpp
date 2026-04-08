#include "madspace/driver/adam_optimizer.h"

#include "madspace/constants.h"

using namespace madspace;

AdamOptimizer::AdamOptimizer(
    const Function& function,
    ContextPtr context,
    double learning_rate,
    LRSchedule schedule,
    std::size_t step_count,
    double beta1,
    double beta2,
    double eps
) :
    _context(context),
    _runtime(build_runtime(function, context)),
    _learning_rate(learning_rate),
    _schedule(schedule),
    _step(0),
    _step_count(step_count),
    _beta1(beta1),
    _beta2(beta2),
    _eps(eps) {
    DevicePtr device = context->device();
    for (auto& [name, value] : function.globals()) {
        Tensor global = context->global(name);
        _parameters.push_back(global);
        _exp_avgs.emplace_back(global.dtype(), global.shape(), device).zero();
        _exp_avg_sqs.emplace_back(global.dtype(), global.shape(), device).zero();
    }
    _input_types.reserve(function.inputs().size());
    for (auto& input : function.inputs()) {
        _input_types.push_back(input.type);
    }
}

TensorVec AdamOptimizer::step(const TensorVec& inputs) {
    double lr = learning_rate();
    ++_step;
    double bias_corr1 = 1 - std::pow(_beta1, _step);
    double bias_corr2 = 1 - std::pow(_beta2, _step);
    double step_size = lr / bias_corr1;
    double bias_corr2_sqrt = std::sqrt(bias_corr2);
    auto [outputs, stored_locals, eval_grad] =
        _runtime->run_with_grad(inputs, std::vector<bool>(inputs.size(), false));
    TensorVec output_grads(outputs.size());
    DevicePtr device = _context->device();
    output_grads.at(0) = Tensor(1.0, device);
    auto [input_grads, global_grads] =
        _runtime->run_backward(output_grads, stored_locals, eval_grad);
    /*device->adam_step(
        global_grads,
        _parameters,
        _exp_avgs,
        _exp_avg_sqs,
        step_size,
        _beta1,
        _beta2,
        _eps,
        bias_corr2_sqrt
    );*/
    return outputs;
}

double AdamOptimizer::learning_rate() const {
    switch (_schedule) {
    case none:
        return _learning_rate;
    case cosine_annealing:
        return 0.5 * _learning_rate * (1 + std::cos(_step * PI / _step_count));
    default:
        throw std::runtime_error("Invalid LR schedule");
    }
}
