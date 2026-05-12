#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"

namespace madspace {

class TwoBodyDecay : public Mapping {
public:
    TwoBodyDecay(bool com);
    std::size_t random_dim() const { return 2; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;

    bool _com;
};

class TwoToTwoParticleScattering : public Mapping {
public:
    TwoToTwoParticleScattering(
        bool com, double invariant_power = 0, double mass = 0, double width = 0
    );

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;

    bool _com;
    Invariant _invariant;
};

} // namespace madspace
