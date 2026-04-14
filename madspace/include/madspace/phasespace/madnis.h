#pragma once

#include <format>
#include <vector>

#include "madspace/phasespace/base.h"
#include "madspace/phasespace/channel_weight_network.h"

namespace madspace {

class MadnisLoss : public FunctionGenerator {
public:
    MadnisLoss(
        const std::vector<std::shared_ptr<FunctionGenerator>>& functions,
        const std::optional<ChannelWeightNetwork>& cwnet
    );

private:
    ValueVec build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::shared_ptr<FunctionGenerator>> _functions;
    std::optional<ChannelWeightNetwork> _cwnet;
};

} // namespace madspace
