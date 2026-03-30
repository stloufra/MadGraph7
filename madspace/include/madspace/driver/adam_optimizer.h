#pragma once

#include "madspace/driver/backend.h"
#include "madspace/driver/context.h"

namespace madspace {

class AdamOptimizer {
public:
    enum LRSchedule {
        none,
        cosine_annealing,
    };

    AdamOptimizer(
        const Function& function,
        ContextPtr context,
        double learning_rate,
        LRSchedule schedule,
        std::size_t step_count = 0,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double eps = 1.e-8
    );
    TensorVec step(const TensorVec& inputs);
    double learning_rate() const;

private:
    ContextPtr _context;
    RuntimePtr _runtime;
    LRSchedule _schedule;
    double _learning_rate;
    std::size_t _step;
    std::size_t _step_count;
    double _beta1;
    double _beta2;
    double _eps;
    TensorVec _parameters;
    TensorVec _exp_avgs;
    TensorVec _exp_avg_sqs;
};

} // namespace madspace
