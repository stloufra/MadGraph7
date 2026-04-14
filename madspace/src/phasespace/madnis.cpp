#include "madspace/phasespace/madnis.h"

using namespace madspace;

MadnisLoss::MadnisLoss(
    const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
    const std::optional<ChannelWeightNetwork>& cwnet
) :
    FunctionGenerator(
        "MadnisLoss",
        [&] {
            TypeVec arg_types;
            for (auto& func : functions) {
                arg_types.insert(
                    arg_types.end(), func->arg_types().begin(), func->arg_types().end()
                );
            }
            return arg_types;
        }(),
        {single_float,
         single_float_array(functions.size()),
         single_float_array(functions.size())}
    ),
    _functions(functions),
    _cwnet(cwnet) {}

ValueVec MadnisLoss::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    std::vector<ValueVec> split_outputs(return_types().size());
    for (std::size_t index = 0, arg_index = 0; auto& func : _functions) {
        std::size_t arg_index_end = arg_index + func->arg_types().size();
        ValueVec func_args(args.begin() + arg_index, args.begin() + arg_index_end);
        fb.set_current_stream(index + 1);
        ValueVec output = func->build_function(fb, func_args);
        for (std::size_t split_out_index = 0; auto& out : output) {
            split_outputs.at(split_out_index).push_back(out);
            ++split_out_index;
        }
        // compute mean
        // compute variance
        // compute loss
        ++index;
        arg_index = arg_index_end;
    }
    fb.set_current_stream(0);
    ValueVec cat_outputs;
    for (auto& output : split_outputs) {
        auto [cat, _] = fb.batch_cat(output);
        cat_outputs.push_back(cat);
    }
    return cat_outputs;
}
