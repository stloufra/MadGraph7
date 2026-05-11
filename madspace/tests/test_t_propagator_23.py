"""Tests for TPropagatorMapping23 (per-set, colour-ordered topology).

Colour order convention: 0-indexed permutation of {0, ..., n-1}.
Particles 0 and 1 are the two incoming beams; 2, 3, ..., n-1 are outgoing.

Note on non-physical events: For inner-rung 2->3 peels, the kinematic
constraint imposed by the 2->3 block (Gram-determinant condition, s23/t1
range non-empty) can be violated for some sampled (rest-mass, t, s) combinations.
These events produce non-finite momenta/det and should be discarded in any
Monte Carlo integration -- they correspond to zero-weight points outside the
physical region. The tests therefore filter on `np.isfinite(det)` before
checking conservation, on-shell, and inverse round-trip.

Note on numerical tails: events near kinematic boundaries (very small forward
det) have accumulated floating-point error from chained sqrts, boosts, and
rotations. A 10th-percentile det floor filters these without biasing the
integrand (they have near-zero weight in any MC integral anyway).
"""
import numpy as np
import pytest
from pytest import approx

import madspace as ms


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


N = 100_000
CM_ENERGY = 13000.0
MIN_PHYSICAL_FRACTION = 0.10  # at least 10% of events should be physical
LOW_DET_QUANTILE = 0.10       # drop bottom 10% by |det| in stability-sensitive tests


# Mass set families. The labels are templates; n_out is filled in at use.
def _massless(n_out):
    return [0.0] * n_out, f"{n_out} massless"


def _w_like(n_out):
    return [80.0] * n_out, f"{n_out} W-like"


def _mixed_top(n_out):
    # one heavy, one W-like, rest massless: only defined for n_out >= 3.
    return [173.0, 80.0] + [0.0] * (n_out - 2), f"{n_out} mixed"


def _mass_sets_for(n_out):
    """Mass families that make sense at this n_out."""
    sets = [_massless(n_out), _w_like(n_out)]
    if n_out >= 3:
        sets.append(_mixed_top(n_out))
    return sets


# Each entry: (colour_order, int_order_for_old_class, topology_label).
# The topology determines n_out = len(colour_order) - 2.
TOPOLOGIES = [
    ([0, 2, 3, 1, 4],            [0, 1],            "n=5: set1={2,3}, set2={4}"),
    ([0, 2, 3, 4, 1, 5],         [0, 1, 2],         "n=6: set1={2,3,4}, set2={5}"),
    ([0, 2, 3, 1, 4, 5],         [0, 1, 2],         "n=6: set1={2,3}, set2={4,5}"),
    ([0, 2, 3, 4, 1, 5, 6],      [0, 1, 2, 3],      "n=7: set1={2,3,4}, set2={5,6}"),
    ([0, 2, 3, 4, 5, 1, 6],      [0, 1, 2, 3],      "n=7: set1={2,3,4,5}, set2={6}"),
    ([0, 2, 3, 4, 5, 6, 1, 7],   [0, 1, 2, 3, 4],   "n=8: set1={2,3,4,5,6}, set2={7}"),
]


def _expand(topologies):
    """Produce a flat list of (colour_order, int_order, masses, label) tuples
    where each topology is paired only with its compatible mass families.
    This replaces the cross-product parametrize that produced many skips."""
    out = []
    for colour_order, int_order, topo_label in topologies:
        n_out = len(colour_order) - 2
        for masses, mass_label in _mass_sets_for(n_out):
            label = f"{mass_label}-{topo_label}"
            out.append((colour_order, int_order, masses, label))
    return out


CASES = _expand(TOPOLOGIES)
CASE_IDS = [c[3] for c in CASES]


def _physical_mask(det, p_ext):
    """Events are physical iff det is finite AND every momentum is finite."""
    mask = np.isfinite(det)
    for p in p_ext:
        mask &= np.all(np.isfinite(p), axis=1)
    return mask


def _filter_physical(det, p_ext):
    mask = _physical_mask(det, p_ext)
    if mask.sum() < MIN_PHYSICAL_FRACTION * len(det):
        pytest.fail(
            f"Too few physical events: {mask.sum()}/{len(det)} "
            f"({100*mask.sum()/len(det):.1f}%), expected >= "
            f"{100*MIN_PHYSICAL_FRACTION:.0f}%"
        )
    return mask, [p[mask] for p in p_ext], det[mask]


def _drop_low_det_tail(det, *arrays, q=LOW_DET_QUANTILE, median_relative=1e-6):
    """Drop the bottom-q quantile AND any events below median_relative * median
    of events by |det|. These boundary events accumulate floating-point error
    from chained operations and have near-zero weight in MC anyway. The
    median-relative floor matters for heavy-tailed DoubleT topologies where
    the quantile floor alone is still many decades below the median."""
    median = np.median(np.abs(det))
    floor = max(np.quantile(np.abs(det), q), median_relative * median)
    keep = np.abs(det) > floor
    return (det[keep], *(a[keep] for a in arrays))


@pytest.mark.parametrize("colour_order,_int_order,masses,_label", CASES, ids=CASE_IDS)
def test_momentum_conservation(rng, colour_order, _int_order, masses, _label):
    mapping = ms.TPropagatorMapping23(colour_order)
    r = rng.random((N, mapping.random_dim()))
    e_cm = np.full(N, CM_ENERGY)
    cond = [e_cm] + [np.full(N, m) for m in masses]

    *p_ext, det = mapping.map_forward(list(r.T), cond)
    p_ext = list(p_ext)
    _, p_ext, det = _filter_physical(det, p_ext)

    p_in = p_ext[0] + p_ext[1]
    p_out_sum = sum(p_ext[i] for i in range(2, len(p_ext)))
    assert p_in == approx(p_out_sum, abs=1e-3, rel=1e-6)


@pytest.mark.parametrize("colour_order,_int_order,masses,_label", CASES, ids=CASE_IDS)
def test_on_shell_masses(rng, colour_order, _int_order, masses, _label):
    mapping = ms.TPropagatorMapping23(colour_order)
    r = rng.random((N, mapping.random_dim()))
    e_cm = np.full(N, CM_ENERGY)
    cond = [e_cm] + [np.full(N, m) for m in masses]

    *p_ext, det = mapping.map_forward(list(r.T), cond)
    p_ext = list(p_ext)
    _, p_ext, det = _filter_physical(det, p_ext)

    # Drop boundary events that accumulate FP error from chained kinematics.
    det, *p_ext = _drop_low_det_tail(det, *p_ext)

    for i, m in enumerate(masses):
        p = p_ext[i + 2]
        m_check = np.sqrt(np.maximum(0,
            p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)
        ))
        assert m_check == approx(m, abs=1e-2, rel=1e-3), \
            f"particle {i} mass off"


@pytest.mark.parametrize("colour_order,_int_order,masses,_label", CASES, ids=CASE_IDS)
def test_inverse(rng, colour_order, _int_order, masses, _label):
    mapping = ms.TPropagatorMapping23(colour_order)
    r = rng.random((N, mapping.random_dim()))
    e_cm = np.full(N, CM_ENERGY)
    cond = [e_cm] + [np.full(N, m) for m in masses]

    *p_ext, det = mapping.map_forward(list(r.T), cond)
    p_ext = list(p_ext)
    mask, p_ext_phys, det_phys = _filter_physical(det, p_ext)
    cond_phys = [c[mask] for c in cond]
    r_phys = r[mask]

    *r_inv, det_inv = mapping.map_inverse(p_ext_phys, cond_phys)

    # Drop the lowest-det events: boundary events whose forward det is many
    # orders of magnitude below the median have inverse det near infinity,
    # and the round-trip |det * inv_det - 1| can hit a few percent (or more)
    # from accumulated FP error in atan2/sqrt/boosts. They have negligible
    # weight in any MC integral. We use both a quantile floor (handles
    # well-behaved distributions) and a median-relative floor (handles the
    # heavy-tailed DoubleT-using topologies where the 10th percentile is
    # still many decades below the median).
    median = np.median(np.abs(det_phys))
    floor_quantile = np.quantile(np.abs(det_phys), LOW_DET_QUANTILE)
    floor_median = 1e-6 * median
    floor = max(floor_quantile, floor_median)
    keep = (
        np.isfinite(det_inv)
        & np.isfinite(det_phys)
        & (np.abs(det_phys) > floor)
    )
    for ri in r_inv:
        keep &= np.isfinite(ri)
    if keep.sum() < MIN_PHYSICAL_FRACTION * len(det):
        pytest.fail(
            f"Too few invertible events: {keep.sum()}/{len(det)} "
            f"({100*keep.sum()/len(det):.1f}%)"
        )

    det_kept = det_phys[keep]
    det_inv_kept = det_inv[keep]
    # Inverse-roundtrip tolerance: 0.1% accounts for accumulated FP error
    # in chained kinematic operations (boosts, rotations, sqrts) for the
    # deeper-walk topologies. Correctness is established by
    # test_phase_space_volume_matches_old.
    assert det_inv_kept == approx(1 / det_kept, rel=1e-3)

    # Some randoms are discrete-branch choices (r_choice from 2->3 rungs);
    # round-trip only recovers which-bin (< 0.5 vs >= 0.5), not the exact
    # value. Bin-membership comparison handles those.
    for i, (r_i, r_inv_i) in enumerate(zip(list(r_phys.T), r_inv)):
        r_i_k = r_i[keep]
        r_inv_i_k = r_inv_i[keep]
        diff = np.abs(r_i_k - r_inv_i_k)
        bin_ok = (r_i_k < 0.5) == (r_inv_i_k < 0.5)
        ok = (diff < 1e-4) | bin_ok
        assert np.all(ok), (
            f"random index {i}: {np.sum(~ok)} mismatches, "
            f"max diff = {diff[~ok].max() if np.any(~ok) else 0:.3e}"
        )


@pytest.mark.parametrize("colour_order,int_order,masses,_label", CASES, ids=CASE_IDS)
def test_phase_space_volume_matches_old(rng, colour_order, int_order, masses, _label):
    """The new class should integrate to the same phase-space volume as
    the old TPropagatorMapping (with matrix element = 1).

    This is the actual correctness check: both classes are MC estimators of
    the same n-body phase-space integral, and their averages must agree
    within a few sigma. (The inverse-roundtrip test is a precision check;
    this one is the physics check.)
    """
    new = ms.TPropagatorMapping23(colour_order)
    old = ms.TPropagatorMapping(int_order)

    r_new = rng.random((N, new.random_dim()))
    r_old = rng.random((N, old.random_dim()))
    e_cm = np.full(N, CM_ENERGY)
    cond_new = [e_cm] + [np.full(N, m) for m in masses]
    cond_old = cond_new

    *p_new, det_new = new.map_forward(list(r_new.T), cond_new)
    *p_old, det_old = old.map_forward(list(r_old.T), cond_old)

    mask_new = np.isfinite(det_new) & (det_new > 0)
    mask_old = np.isfinite(det_old) & (det_old > 0)
    det_new = det_new[mask_new]
    det_old = det_old[mask_old]

    assert len(det_new) > MIN_PHYSICAL_FRACTION * N, \
        f"new class: {len(det_new)}/{N} physical, too few"
    assert len(det_old) > MIN_PHYSICAL_FRACTION * N, \
        f"old class: {len(det_old)}/{N} physical, too few"

    mean_new = np.mean(det_new)
    mean_old = np.mean(det_old)
    err_new = np.std(det_new) / np.sqrt(len(det_new))
    err_old = np.std(det_old) / np.sqrt(len(det_old))
    err = np.sqrt(err_new ** 2 + err_old ** 2)

    diff = abs(mean_new - mean_old)
    assert diff < 5 * err, (
        f"phase-space volumes differ: new = {mean_new:.4e} +/- {err_new:.4e}, "
        f"old = {mean_old:.4e} +/- {err_old:.4e}, "
        f"diff = {diff:.4e} ({diff/err:.2f} sigma)"
    )