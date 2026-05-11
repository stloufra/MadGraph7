#include "madspace/phasespace/t_propagator_mapping_23.hpp"

#include <algorithm>
#include <stdexcept>

#include "madspace/util.hpp"

using namespace madspace;

namespace {

// Cyclically rotate the colour order so that particle 0 is first, then
// split the rest into two sets: particles strictly between 0 and 1 in the
// cyclic order go into set1, particles after 1 (wrapping around) go into
// set2. Both sets are returned as 0-indexed outgoing-particle indices,
// i.e. particle p (0-indexed external) becomes (p - 2) in our 0-indexed
// convention for outgoing particles. (Particles 0 and 1 are the beams
// and not in either set.)
std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
split_sets_from_colour_order(const std::vector<std::size_t>& colour_order) {
    std::size_t n = colour_order.size();
    if (n < 4) {
        throw std::invalid_argument(
            "TPropagatorMapping23 requires at least 4 particles (2 beams + 2 outgoing)"
        );
    }
    // Find index of particle 0 and rotate
    auto it = std::find(colour_order.begin(), colour_order.end(), 0u);
    if (it == colour_order.end()) {
        throw std::invalid_argument("colour_order must contain particle 0");
    }
    std::size_t i0 = std::distance(colour_order.begin(), it);
    std::vector<std::size_t> rotated;
    rotated.reserve(n);
    for (std::size_t k = 0; k < n; ++k) {
        rotated.push_back(colour_order[(i0 + k) % n]);
    }
    // rotated[0] is now particle 0. Find particle 1.
    auto it1 = std::find(rotated.begin(), rotated.end(), 1u);
    if (it1 == rotated.end()) {
        throw std::invalid_argument("colour_order must contain particle 1");
    }
    std::size_t i1 = std::distance(rotated.begin(), it1);
    // set1 = rotated[1..i1-1], set2 = rotated[i1+1..n-1]
    std::vector<std::size_t> set1, set2;
    for (std::size_t k = 1; k < i1; ++k) {
        std::size_t p = rotated[k];  // 0-indexed external
        if (p <= 1) throw std::invalid_argument("invalid colour_order");
        set1.push_back(p - 2);       // 0-indexed outgoing
    }
    for (std::size_t k = i1 + 1; k < n; ++k) {
        std::size_t p = rotated[k];
        if (p <= 1) throw std::invalid_argument("invalid colour_order");
        set2.push_back(p - 2);
    }
    if (set1.empty() || set2.empty()) {
        throw std::invalid_argument(
            "TPropagatorMapping23 requires both sets to be non-empty "
            "(particles 0 and 1 must not be adjacent in the colour order)"
        );
    }
    return {set1, set2};
}

// Number of intermediate rest masses to sample for a walk of size k:
//   max(0, k - 2)
std::size_t n_intermediate_masses_for_set_size(std::size_t k) {
    return (k >= 2) ? (k - 2) : 0;
}

// Number of randoms a walk of size k consumes for its t-channel-block calls
// (excluding mass samples).
std::size_t n_block_randoms_for_set_size(std::size_t k) {
    if (k <= 1) return 0;
    // First peel: 2->2 LAB (2 randoms); each subsequent peel: 2->3 (3 randoms).
    return 2 + 3 * (k - 2);
}

}  // namespace

TPropagatorMapping23::TPropagatorMapping23(
    const std::vector<std::size_t>& colour_order,
    double t_invariant_power,
    double s_invariant_power
) :
    Mapping(
        "TPropagatorMapping23",
        [&] {
            auto [s1, s2] = split_sets_from_colour_order(colour_order);
            bool use_double_t = (s1.size() == 1) != (s2.size() == 1);
            // When DoubleT is used: neither set-mass is sampled (the
            // single-particle side has fixed mass, the multi-particle side's
            // mass is derived by DoubleT), and the central block consumes
            // 3 randoms (r_phi, r_t1, r_t2). Otherwise: 1 set-mass per
            // multi-particle side, plus 2 randoms for the central 2->2.
            std::size_t n_set_masses = use_double_t
                ? 0u
                : (s1.size() >= 2 ? 1u : 0u) + (s2.size() >= 2 ? 1u : 0u);
            std::size_t n_central = use_double_t ? 3u : 2u;
            std::size_t n_intermediate_masses =
                n_intermediate_masses_for_set_size(s1.size())
              + n_intermediate_masses_for_set_size(s2.size());
            std::size_t n_walk =
                n_block_randoms_for_set_size(s1.size())
              + n_block_randoms_for_set_size(s2.size());
            std::size_t total = n_set_masses + n_intermediate_masses
                              + n_central + n_walk;
            NamedVector<Type> input_types;
            for (std::size_t i = 0; i < total; ++i) {
                input_types.push_back(std::format("random{}", i), batch_float);
            }
            return input_types;
        }(),
        [&] {
            std::size_t n_out = colour_order.size() - 2;
            NamedVector<Type> output_types;
            for (std::size_t i = 0; i < n_out + 2; ++i) {
                output_types.push_back(std::format("momentum{}", i), batch_four_vec);
            }
            return output_types;
        }(),
        [&] {
            std::size_t n_out = colour_order.size() - 2;
            NamedVector<Type> cond_types{{"com_energy", batch_float}};
            for (std::size_t i = 0; i < n_out; ++i) {
                cond_types.push_back(std::format("mass{}", i), batch_float);
            }
            return cond_types;
        }()
    ),
    _n_out(colour_order.size() - 2),
    _com_scattering(true, t_invariant_power),
    _lab_scattering(false, t_invariant_power),
    _two_to_three(t_invariant_power, 0., 0., s_invariant_power, 0., 0.),
    _double_t(t_invariant_power, 0., 0., t_invariant_power, 0., 0.) {
    auto [s1, s2] = split_sets_from_colour_order(colour_order);
    _set1 = s1;
    _set2 = s2;
    _use_double_t = (s1.size() == 1) != (s2.size() == 1);
    std::size_t n_set_masses = _use_double_t
        ? 0u
        : (s1.size() >= 2 ? 1u : 0u) + (s2.size() >= 2 ? 1u : 0u);
    std::size_t n_central = _use_double_t ? 3u : 2u;
    std::size_t n_intermediate_masses =
        n_intermediate_masses_for_set_size(s1.size())
      + n_intermediate_masses_for_set_size(s2.size());
    std::size_t n_walk =
        n_block_randoms_for_set_size(s1.size())
      + n_block_randoms_for_set_size(s2.size());
    _random_dim = n_set_masses + n_intermediate_masses + n_central + n_walk;
}

// === Helpers used in both build_forward and build_inverse ===
namespace {

// Sum of masses of particles at indices `idxs` (0-indexed outgoing) in m_out.
Value sum_of_masses(
    FunctionBuilder& fb, const ValueVec& m_out, const std::vector<std::size_t>& idxs
) {
    Value s = m_out.at(idxs[0]);
    for (std::size_t k = 1; k < idxs.size(); ++k) {
        s = fb.add(s, m_out.at(idxs[k]));
    }
    return s;
}

}  // namespace

Mapping::Result TPropagatorMapping23::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    auto r = inputs.begin();
    auto next_random = [&]() { return *(r++); };
    ValueVec dets;

    // ============================================================
    // Phase 1a: pre-sample set composite masses.
    // When _use_double_t: neither set mass is sampled here. The single-
    // particle side has fixed mass; the multi-particle side's mass is
    // derived later from the DoubleT central block.
    // Otherwise: sample multi-particle set masses, fix single-particle ones.
    // ============================================================
    Value m_set1, m_set2;
    Value mass_sum_set1 = sum_of_masses(fb, m_out, _set1);
    Value mass_sum_set2 = sum_of_masses(fb, m_out, _set2);
    if (_use_double_t) {
        // Both set masses set later (multi-side from DoubleT output,
        // single-side fixed).
        if (_set1.size() == 1) m_set1 = m_out.at(_set1[0]);
        if (_set2.size() == 1) m_set2 = m_out.at(_set2[0]);
        // The multi-side m_set will be filled in after the DoubleT call,
        // before sampling intermediate rest masses.
    } else {
        if (_set1.size() >= 2) {
            auto s_min = fb.square(mass_sum_set1);
            auto s_max = fb.square(fb.sub(e_cm, mass_sum_set2));
            auto res = _uniform_invariant.build_forward(fb, {next_random()}, {s_min, s_max});
            m_set1 = fb.sqrt(res["invariant"]);
            dets.push_back(res["det"]);
        } else {
            m_set1 = m_out.at(_set1[0]);
        }
        if (_set2.size() >= 2) {
            auto s_min = fb.square(mass_sum_set2);
            auto s_max = fb.square(fb.sub(e_cm, m_set1));
            auto res = _uniform_invariant.build_forward(fb, {next_random()}, {s_min, s_max});
            m_set2 = fb.sqrt(res["invariant"]);
            dets.push_back(res["det"]);
        } else {
            m_set2 = m_out.at(_set2[0]);
        }
    }

    // ============================================================
    // Phase 2: central block producing (P_set1, P_set2).
    // ============================================================
    auto [pa, pb] = fb.com_p_in(e_cm);
    Value P_set1, P_set2;
    if (_use_double_t) {
        // DoubleT: pa, pb -> (single, recoil). The single particle's mass
        // is m_single (fixed); the recoil mass must be >= mir_min (sum of
        // recoil-side outgoing masses).
        bool single_is_set1 = (_set1.size() == 1);
        Value m_single = single_is_set1 ? m_set1 : m_set2;
        Value mir_min = single_is_set1 ? mass_sum_set2 : mass_sum_set1;
        auto central = _double_t.build_forward(
            fb,
            {next_random(), next_random(), next_random()},
            {pa, pb, m_single, mir_min}
        );
        Value p_single = central.at(0);   // mass = m_single
        Value p_recoil = central.at(1);   // recoil; mass derived from t1, t2
        dets.push_back(central["det"]);
        if (single_is_set1) {
            P_set1 = p_single;
            P_set2 = p_recoil;
            // m_set1 already set above. Derive m_set2 from p_recoil for
            // downstream intermediate-mass sampling. invariants_from_momenta
            // returns m^2 for the summed momenta indicated by the factor row.
            nested_vector2<double> factors_recoil{{1.0}};
            auto m2 = fb.unstack(
                fb.invariants_from_momenta(fb.stack({p_recoil}), factors_recoil)
            ).at(0);
            m_set2 = fb.sqrt(m2);
        } else {
            P_set2 = p_single;
            P_set1 = p_recoil;
            nested_vector2<double> factors_recoil{{1.0}};
            auto m2 = fb.unstack(
                fb.invariants_from_momenta(fb.stack({p_recoil}), factors_recoil)
            ).at(0);
            m_set1 = fb.sqrt(m2);
        }
    } else {
        auto central = _com_scattering.build_forward(
            fb,
            {next_random(), next_random(), m_set1, m_set2},
            {pa, pb}
        );
        P_set1 = central.at(0);
        P_set2 = central.at(1);
        dets.push_back(central["det"]);
    }

    // The "other beam after central block" for each set's walk:
    // by 4-momentum conservation, p_other_beam - P_other_set is what is
    // (kinematically) left on that beam line after the central block emitted
    // the other set.
    Value R_b_for_set1 = fb.sub(pb, P_set2);
    Value R_b_for_set2 = fb.sub(pa, P_set1);

    // ============================================================
    // Phase 1b: pre-sample intermediate rest masses for each walk.
    // For a walk peeling particles set[0], set[1], ..., set[k-1]:
    //   - mass of rest after peeling set[0]   = m_rest_1  (sampled)
    //   - mass of rest after peeling set[0..1] = m_rest_2  (sampled)
    //   - ...
    //   - mass of rest after peeling set[0..k-3] = m_rest_{k-2}  (sampled)
    //   - mass of rest after peeling set[0..k-2] = m of set[k-1]  (fixed, residual)
    // Bounds:
    //   m_min = sum of masses of [set[j+1], ..., set[k-1]]
    //   m_max = m_set - sum of masses of [set[0], ..., set[j]]
    // Note: we sample these AFTER the central block, because for DoubleT
    // we don't know m_set (multi-side) until then.
    // ============================================================
    auto sample_intermediate_masses =
        [&](const std::vector<std::size_t>& s, Value m_set) -> ValueVec {
        std::size_t k = s.size();
        ValueVec res;
        if (k <= 2) {
            return res;  // no intermediate masses needed
        }
        // The walk peels particles s[0], s[1], ..., s[k-2] (s[k-1] is the
        // final residual). After peeling s[0..j], the "rest" has mass
        // m_rest_{j+1}. The chain is monotonic:
        //   m_set > m_rest_1 > m_rest_2 > ... > m_rest_{k-2} > mass(s[k-1])
        // We sample m_rest_{j+1} from
        //   m_min = sum of masses [s[j+1]..s[k-1]]  (kinematic lower bound)
        //   m_max = m_rest_j - mass[s[j]]            (monotonicity + peel-off room)
        // where m_rest_0 = m_set.
        // Sampling independently from [m_min, m_set - sum_of_already_peeled]
        // would over-count by (k-2)! because it ignores the ordering.
        Value prev_mass = m_set;
        for (std::size_t j = 0; j < k - 2; ++j) {
            Value m_min = m_out.at(s[j + 1]);
            for (std::size_t i = j + 2; i < k; ++i) {
                m_min = fb.add(m_min, m_out.at(s[i]));
            }
            auto s_min = fb.square(m_min);
            auto s_max = fb.square(fb.sub(prev_mass, m_out.at(s[j])));
            auto r = _uniform_invariant.build_forward(fb, {next_random()}, {s_min, s_max});
            Value m_rest = fb.sqrt(r["invariant"]);
            res.push_back(m_rest);
            dets.push_back(r["det"]);
            prev_mass = m_rest;
        }
        return res;
    };

    ValueVec rest_masses_set1 = sample_intermediate_masses(_set1, m_set1);
    ValueVec rest_masses_set2 = sample_intermediate_masses(_set2, m_set2);

    // ============================================================
    // Phase 3: peel-off walks
    // ============================================================
    ValueVec p_out(_n_out);  // 0-indexed by outgoing particle index

    auto walk = [&](const std::vector<std::size_t>& s,
                    Value P_set,
                    Value R_b,
                    const ValueVec& rest_masses) {
        std::size_t k = s.size();
        if (k == 1) {
            p_out[s[0]] = P_set;
            return;
        }
        Value R_a = fb.sub(P_set, R_b);
        Value im1;
        bool first = true;
        for (std::size_t j = 0; j < k - 1; ++j) {
            // Peel particle s[j]. The "new rest" after this peel has either
            // mass = rest_masses[j] (intermediate) or mass = m_out[s[k-1]]
            // (the final residual, when j == k-2).
            Value m_rest = (j < k - 2) ? rest_masses[j] : m_out.at(s[k - 1]);
            Value m_peel = m_out.at(s[j]);
            // Block input: (R_b, R_a). Block output: p1_out + p2_out = R_b + R_a,
            // where p1_out has mass m_rest (the chain carrier, sits on the pa=R_b
            // side per the block's convention) and p2_out has mass m_peel (the
            // newly peeled particle, sits on the pb=R_a side -- which is what we
            // want since R_a is the "active" leg that emits).
            // R_a is updated as R_a -= peeled. R_b stays constant.
            //
            // For the 2->3 block, the kernel internally computes
            // p_12 = pa + pb - p_3, where p_3 = im1 (previous peel on this side).
            // We want p_12 = remaining-system-left-to-produce = R_a + R_b
            // (using R_a *after* the previous peel was subtracted from it).
            // Since the kernel subtracts im1 internally, and im1 is already
            // subtracted from R_a in our bookkeeping, we restore it: pass
            // pb = R_a + im1, so p_12 = R_b + (R_a + im1) - im1 = R_b + R_a. ✓
            if (first) {
                auto ks = _lab_scattering.build_forward(
                    fb,
                    {next_random(), next_random(), m_rest, m_peel},
                    {R_b, R_a}
                );
                Value peeled = ks.at(1);
                p_out[s[j]] = peeled;
                R_a = fb.sub(R_a, peeled);
                im1 = peeled;
                dets.push_back(ks["det"]);
                first = false;
            } else {
                Value pb_for_block = fb.add(R_a, im1);
                auto ks = _two_to_three.build_forward(
                    fb,
                    {next_random(), next_random(), next_random(), m_rest, m_peel},
                    {R_b, pb_for_block, im1}
                );
                Value peeled = ks.at(1);
                p_out[s[j]] = peeled;
                R_a = fb.sub(R_a, peeled);
                im1 = peeled;
                dets.push_back(ks["det"]);
            }
        }
        // Last particle = R_a + R_b (= what remains after all explicit peels).
        p_out[s[k - 1]] = fb.add(R_a, R_b);
    };

    walk(_set1, P_set1, R_b_for_set1, rest_masses_set1);
    walk(_set2, P_set2, R_b_for_set2, rest_masses_set2);

    // Assemble outputs: momentum0, momentum1 = beams; momentum_{2+i} = outgoing i.
    ValueVec p_ext;
    p_ext.reserve(_n_out + 2);
    p_ext.push_back(pa);
    p_ext.push_back(pb);
    for (std::size_t i = 0; i < _n_out; ++i) {
        p_ext.push_back(p_out[i]);
    }

    return {{output_types().keys(), p_ext}, fb.product(dets)};
}

Mapping::Result TPropagatorMapping23::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    ValueVec random_out;
    ValueVec dets;

    // Inputs: pa, pb, p_out[0], p_out[1], ..., p_out[n_out-1]
    Value pa = inputs.at(0);
    Value pb = inputs.at(1);
    auto p_outgoing = [&](std::size_t i) { return inputs.at(2 + i); };

    // ============================================================
    // Build a factor matrix to extract all needed invariant masses in
    // one shot. Rows (in order):
    //   row 0:                m^2(P_set1)            (if set1.size() >= 2)
    //   row 1:                m^2(P_set2)            (if set2.size() >= 2)
    //   then for set1 if size >= 3:
    //     k_1 - 2 rows, each summing momenta s1[j+1..k_1-1] for j=0..k_1-3
    //   then for set2 if size >= 3: similarly
    // Column = index into inputs (n_out + 2 columns).
    // We record the layout so we can pull invariants out by name afterwards.
    // ============================================================
    nested_vector2<double> invariant_factors;
    auto n_inputs = _n_out + 2;
    auto factor_set = [&](const std::vector<std::size_t>& idxs) {
        std::vector<double> row(n_inputs, 0.0);
        for (auto idx : idxs) row.at(idx + 2) = 1.0;
        return row;
    };
    // Track positions in the stacked invariants output:
    int idx_m2_set1 = -1, idx_m2_set2 = -1;
    int idx_rest_set1_start = -1, idx_rest_set2_start = -1;
    if (_set1.size() >= 2) {
        idx_m2_set1 = invariant_factors.size();
        invariant_factors.push_back(factor_set(_set1));
    }
    if (_set2.size() >= 2) {
        idx_m2_set2 = invariant_factors.size();
        invariant_factors.push_back(factor_set(_set2));
    }
    if (_set1.size() >= 3) {
        idx_rest_set1_start = invariant_factors.size();
        for (std::size_t j = 0; j < _set1.size() - 2; ++j) {
            std::vector<std::size_t> sub(_set1.begin() + j + 1, _set1.end());
            invariant_factors.push_back(factor_set(sub));
        }
    }
    if (_set2.size() >= 3) {
        idx_rest_set2_start = invariant_factors.size();
        for (std::size_t j = 0; j < _set2.size() - 2; ++j) {
            std::vector<std::size_t> sub(_set2.begin() + j + 1, _set2.end());
            invariant_factors.push_back(factor_set(sub));
        }
    }
    ValueVec invariants;
    if (!invariant_factors.empty()) {
        invariants = fb.unstack(
            fb.invariants_from_momenta(fb.stack(inputs.values()), invariant_factors)
        );
    }

    // ============================================================
    // Recover composites P_set1, P_set2 from the momenta.
    // ============================================================
    auto sum_momenta = [&](const std::vector<std::size_t>& s) -> Value {
        Value p = p_outgoing(s[0]);
        for (std::size_t k = 1; k < s.size(); ++k) {
            p = fb.add(p, p_outgoing(s[k]));
        }
        return p;
    };
    Value P_set1 = sum_momenta(_set1);
    Value P_set2 = sum_momenta(_set2);

    // ============================================================
    // Phase 1a inverse: recover set-mass randoms (skipped in DoubleT case).
    // Even when randoms are skipped, we still need m_set1/m_set2 for the
    // Phase 1b intermediate-mass-bound calculation; those come from the
    // invariants_from_momenta result for multi-particle sides.
    // ============================================================
    Value m_set1, m_set2;
    Value mass_sum_set1 = sum_of_masses(fb, m_out, _set1);
    Value mass_sum_set2 = sum_of_masses(fb, m_out, _set2);
    if (_use_double_t) {
        // No set-mass randoms to recover; just set m_set1/m_set2.
        if (_set1.size() == 1) {
            m_set1 = m_out.at(_set1[0]);
            m_set2 = fb.sqrt(invariants.at(idx_m2_set2));
        } else {
            m_set2 = m_out.at(_set2[0]);
            m_set1 = fb.sqrt(invariants.at(idx_m2_set1));
        }
    } else {
        if (_set1.size() >= 2) {
            auto s_min = fb.square(mass_sum_set1);
            auto s_max = fb.square(fb.sub(e_cm, mass_sum_set2));
            auto m2_set1 = invariants.at(idx_m2_set1);
            auto res = _uniform_invariant.build_inverse(fb, {m2_set1}, {s_min, s_max});
            random_out.push_back(res["random"]);
            dets.push_back(res["det"]);
            m_set1 = fb.sqrt(m2_set1);
        } else {
            m_set1 = m_out.at(_set1[0]);
        }
        if (_set2.size() >= 2) {
            auto s_min = fb.square(mass_sum_set2);
            auto s_max = fb.square(fb.sub(e_cm, m_set1));
            auto m2_set2 = invariants.at(idx_m2_set2);
            auto res = _uniform_invariant.build_inverse(fb, {m2_set2}, {s_min, s_max});
            random_out.push_back(res["random"]);
            dets.push_back(res["det"]);
            m_set2 = fb.sqrt(m2_set2);
        } else {
            m_set2 = m_out.at(_set2[0]);
        }
    }

    // ============================================================
    // Phase 2 inverse: central block from (pa, pb). Comes BEFORE Phase 1b
    // because the forward emits central randoms before intermediate-mass
    // randoms (so the multi-side m_set is known by Phase 1b).
    // ============================================================
    if (_use_double_t) {
        bool single_is_set1 = (_set1.size() == 1);
        Value p_single = single_is_set1 ? P_set1 : P_set2;
        Value p_recoil = single_is_set1 ? P_set2 : P_set1;
        Value m_single = single_is_set1 ? m_set1 : m_set2;
        Value mir_min = single_is_set1 ? mass_sum_set2 : mass_sum_set1;
        auto central = _double_t.build_inverse(
            fb,
            {p_single, p_recoil},
            {pa, pb, m_single, mir_min}
        );
        // DoubleT inverse returns (r_phi, r_t1, r_t2, det). m1 is no longer
        // an output (it's a condition now).
        random_out.push_back(central.at(0));
        random_out.push_back(central.at(1));
        random_out.push_back(central.at(2));
        dets.push_back(central["det"]);
    } else {
        auto central = _com_scattering.build_inverse(
            fb,
            {P_set1, P_set2},
            {pa, pb}
        );
        random_out.push_back(central.at(0));
        random_out.push_back(central.at(1));
        dets.push_back(central["det"]);
    }

    // ============================================================
    // Phase 1b inverse: recover intermediate rest-mass randoms
    // ============================================================
    auto recover_intermediate_masses =
        [&](const std::vector<std::size_t>& s, Value m_set, int idx_start) {
        std::size_t k = s.size();
        if (k <= 2) return;
        // Mirror the forward chaining: prev_mass starts at m_set, then becomes
        // the (sqrt of the) actual recovered m2 at each step.
        Value prev_mass = m_set;
        for (std::size_t j = 0; j < k - 2; ++j) {
            Value m_min = m_out.at(s[j + 1]);
            for (std::size_t i = j + 2; i < k; ++i) {
                m_min = fb.add(m_min, m_out.at(s[i]));
            }
            auto s_min = fb.square(m_min);
            auto s_max = fb.square(fb.sub(prev_mass, m_out.at(s[j])));
            auto m2 = invariants.at(idx_start + j);
            auto res = _uniform_invariant.build_inverse(fb, {m2}, {s_min, s_max});
            random_out.push_back(res["random"]);
            dets.push_back(res["det"]);
            prev_mass = fb.sqrt(m2);
        }
    };

    recover_intermediate_masses(_set1, m_set1, idx_rest_set1_start);
    recover_intermediate_masses(_set2, m_set2, idx_rest_set2_start);

    Value R_b_for_set1 = fb.sub(pb, P_set2);
    Value R_b_for_set2 = fb.sub(pa, P_set1);

    // ============================================================
    // Phase 3 inverse: peel-off walks
    // ============================================================
    auto walk_inverse = [&](const std::vector<std::size_t>& s,
                            Value P_set,
                            Value R_b) {
        std::size_t k = s.size();
        if (k == 1) return;
        Value R_a = fb.sub(P_set, R_b);
        Value im1;
        bool first = true;
        for (std::size_t j = 0; j < k - 1; ++j) {
            Value peeled = p_outgoing(s[j]);
            // The block at this rung had inputs (R_a, R_b) and produced
            // (p1_out, p2_out) where p1_out has mass m_rest (carrier of
            // remaining) and p2_out = peeled. By conservation,
            // p1_out = R_a + R_b - peeled.
            Value p1_out = fb.sub(fb.add(R_a, R_b), peeled);
            if (first) {
                auto rs = _lab_scattering.build_inverse(
                    fb,
                    {p1_out, peeled},
                    {R_b, R_a}
                );
                random_out.push_back(rs.at(0));
                random_out.push_back(rs.at(1));
                dets.push_back(rs["det"]);
                first = false;
            } else {
                // Mirror the forward's pb restoration: the 2->3 kernel
                // internally subtracts p_3 = im1, so we pass pb = R_a + im1
                // to get p_12 = R_b + R_a (the remaining-to-produce system).
                Value pb_for_block = fb.add(R_a, im1);
                auto rs = _two_to_three.build_inverse(
                    fb,
                    {p1_out, peeled},
                    {R_b, pb_for_block, im1}
                );
                random_out.push_back(rs.at(0));
                random_out.push_back(rs.at(1));
                random_out.push_back(rs.at(2));
                dets.push_back(rs["det"]);
            }
            R_a = fb.sub(R_a, peeled);
            im1 = peeled;
        }
    };

    walk_inverse(_set1, P_set1, R_b_for_set1);
    walk_inverse(_set2, P_set2, R_b_for_set2);

    return {{input_types().keys(), random_out}, fb.product(dets)};
}
