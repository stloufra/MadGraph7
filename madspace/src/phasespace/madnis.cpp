#include "madspace/phasespace/madnis.h"

using namespace madspace;

MadnisLoss::MadnisLoss(
    const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
    const std::optional<ChannelWeightNetwork>& cwnet
) :
    FunctionGenerator(
        "MadnisLoss",
        [&] {
            NamedVector<Type> arg_types;
            for (auto& func : functions) {
                arg_types.insert_back(func->arg_types());
            }
            return arg_types;
        }(),
        {{"loss", single_float},
         {"means", single_float_array(functions.size())},
         {"variances", single_float_array(functions.size())}}
    ),
    _functions(functions),
    _cwnet(cwnet) {}

NamedVector<Value> MadnisLoss::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    std::vector<ValueVec> split_outputs(return_types().size());
    for (std::size_t index = 0, arg_index = 0; auto& func : _functions) {
        std::size_t arg_index_end = arg_index + func->arg_types().size();
        ValueVec func_args(args.begin() + arg_index, args.begin() + arg_index_end);
        fb.set_current_stream(index + 1);
        auto output = func->build_function(fb, func_args);
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
    return {};
}
