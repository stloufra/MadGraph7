# Merge regression investigation notes

Branch: `claude/feat-madmatrix-aloha-merge`. Investigations requested: log
findings + propose actions, **no fix applied yet** (user decides later).

---

## 1. `test_oneloop_reweighting` (tests/acceptance_tests/test_cmd_reweight.py:122)

### STATUS: RESOLVED
Applied the proposed fix: `get_MEcmd` (test_cmd_reweight.py:95) now uses
`output madevent %s` instead of bare `output %s`. The loop reweight machinery
was never the problem (see analysis below); only the host MadEvent directory
needed pinning.

### Symptom
```
FileNotFoundError: [Errno 2] No such file or directory:
  '.../MGPROC/Events/run_01'
  at get_MEcmd(), test_cmd_reweight.py:98  os.mkdir(.../Events/run_01)
```

### Root cause
NOT in the reweight/loop machinery. The failure is in the **shared setup
helper** `get_MEcmd` (test_cmd_reweight.py:89-103), line 95:

```python
mycmd.run_cmd('import model sm; generate e+ e- > mu+ mu-; output %s' % self.run_dir)
```

This is a **bare `output`**, which since merge commit `83c090d70`
("Set default output mode to mg7") defaults to **mg7 / standalone_mg7
(MadMatrix)** instead of madevent. The mg7 directory layout has no `Events/`
subdir (it is a cudacpp/madmatrix tree: src/, SubProcesses/, lib/, Cards/,
bin/), so the subsequent `os.mkdir(.../Events/run_01)` (line 98) fails, and
`MECmd.MadEventCmdShell(me_dir=self.run_dir)` (line 101) would also reject it.
Verified from the run log: `ProcessExporterMadMatrix` exports the
`e+ e- > mu+ mu-` process — confirming bare output -> MadMatrix.

`get_MEcmd` is shared by several reweight tests, so this breaks all of them,
not only the one-loop one.

### On the loop concern ("loops not supported via standalone_mg7")
The loop reweighting itself is **unaffected** by the default-mode change.
`ReweightInterface` does not use the global default output mode; it has a fixed
class attribute `sa_class = 'standalone_rw'` (reweight_interface.py:70) and uses
`standalone_rw` for the tree part and `standalone_rw` / `[virt=QCD]` for the
virtual part (reweight_interface.py:1945/1948/2055/2265). So the loop ME
generation never routes through standalone_mg7.

### Verification
Temporarily patched ONLY get_MEcmd line 95 `output %s` -> `output madevent %s`
(reverted afterwards) and ran the full test:
- Result: **PASS** (`Ran 1 test in 182.296s OK`).
- Log shows correct loop path: `REWEIGHT: RETRY with generate p p > h j
  [virt=QCD]`, final `rwgt_1 : 13.664 +- 0.202 pb` — matches the reference
  solutions.

### Proposed action (NOT applied)
Fix the **setup helper**, not the test, and it is a one-line mode pin:
- `test_cmd_reweight.py:95` (in `get_MEcmd`): `output %s` -> `output madevent %s`.
- Note `get_aMCcmd` (line 112) already does `generate u u~ > mu+ mu- [QCD]; output %s`;
  that one is an NLO process and bare output auto-routes to aMC@NLO (see #2), so
  it likely does NOT need a change — but it should be checked if any test using
  `get_aMCcmd` is red.
This is consistent with the user's guidance: the loop part is fine; the only
change needed is making the *host* MadEvent directory actually be a madevent
output. It is not a change to the test's own (reweight) behaviour.

---

## 2. `test_amcatnlo_from_file` (tests/acceptance_tests/test_cmd_amcatnlo.py:759)

Input script: `tests/input_files/test_amcatnlo`
```
set automatic_html_opening False --no_save
generate p p > e+ ve QED=2 QCD=0 [QCD]
output
launch -p
set nevents 100
set MZ 80
done
```

### Finding: could NOT reproduce a deterministic failure — it PASSES
- `python3 tests/test_manager.py test_amcatnlo_from_file -pA -t0` ->
  `Ran 1 test in 45.439s OK`.
- Summary parsed correctly: `Run at p-p collider (6500.0 + 6500.0 GeV)`,
  `Number of events generated: 100`, `Total cross section: 6.670e+03 +- 3.1e+01 pb`.
  Assertion is `assertAlmostEqual(6675.0, xsec, delta=50)` -> 6670 is inside
  [6625, 6725]. PASS.

### The bare-output / mg7 default does NOT affect this test
Confirmed by direct probe: for the NLO process `p p > e+ ve [QCD]`, a bare
`output` (no path) still routes to **aMC@NLO** and produces
`PROCNLO_loop_sm_0` (full aMC@NLO tree: bin/, Cards/, Events/, FixedOrderAnalysis/,
HTML/, lib/, makefile). The NLO auto-detection overrides the mg7 default. So the
merge's default-mode change is NOT a regression source here — matching the
user's expectation that this test "should not have been impacted by the merge".

### Determinism check: it is reproducible, not flaky
Ran the test twice. Both runs produced the **identical** Summary cross section:
- Run 1: `Total cross section: 6.670e+03 +- 3.1e+01 pb` -> PASS
- Run 2: `Total cross section: 6.670e+03 +- 3.1e+01 pb` -> PASS
Even though the run_card carries `iseed = 0`, aMC@NLO resolves a deterministic
seed for a fresh output, so repeated fresh runs reproduce the same value. The
flakiness hypothesis is therefore NOT supported by the data — the test is stable
and green locally on f4b2ff8a9.

### Proposed lines of action (NOT applied — user to choose)
1. **I cannot reproduce any failure.** The test passes deterministically with
   the correct NLO routing (aMC@NLO, not mg7). To go further I need the actual
   failing output: please share the CI log / the xsec value or error message
   from the run you saw fail.
2. Things to check on the failing environment when that log is available:
   - Is the failure the same `unittest.debug` AttributeError seen when invoking
     via `python3 -m unittest` directly (a harness-invocation artefact, fixed by
     running through `tests/test_manager.py`)? That is NOT a merge regression.
   - Is it a cross-section just outside [6625, 6725] (would point at a
     parameter/scale/PDF difference in that environment, e.g. a different
     LHAPDF set), or a parse error (Summary format), or a crash during
     `launch -p` (compilation/MadLoop)?
   - State-leakage when run inside the full suite vs in isolation (passes in
     isolation here).

### Bottom line
No reproducible evidence the merge regressed this test. The NLO process's bare
`output` correctly routes to aMC@NLO (not mg7), and the result is stable and in
tolerance across repeated runs. Need the concrete failing output to investigate
any real regression.

---

## 3. `test_standalone_flavor_mask` (tests/acceptance_tests/test_cmd.py:864)

### STATUS: RESOLVED
Rewrote the C++ white-box patching to the merged standalone_cpp architecture:
the matrix element is now evaluated by index via `process.sigmaKin(iflav)`,
which reads CPPProcess's internal `flavor_table` and the per-flavor bookkeeping
arrays sized by `nflavors`. The two non-representative flavours are now injected
by extending that internal `flavor_table` (CPPProcess.cc) + `nflavors`
(CPPProcess.h) and `maxflavor`/`pdg_arr` (check_sa.cpp), instead of the old
check_sa-local `flavor_arr`. A multi-line-safe array-extender helper was added.
Verified: the test passes (both Fortran and C++ backends); s c~ > s c~ reproduces
the d u~ > d u~ value (8.5706e-3), s c~ > c c~ vanishes, and the runtime masks are
partial (911/56 vs 4095/255) for the known flavour and all-on for the lookup miss.

### Symptom (run on f4b2ff8a9+)
- **Fortran standalone backend: PASSES** (the first `assert_backend`, line 1002).
- **C++ standalone_cpp backend: FAILS** (line 1055):
```
AssertionError: (3, -4, 4, -4) not found in { ... (3, -4, 3, -4): 0.0 ... }
```
i.e. for the patched-in non-representative flavors:
  - `s c~ > s c~` (PDG 3 -4 3 -4) returns **0.0** instead of the reference
    `d u~ > d u~` value (~0.0086), and
  - `s c~ > c c~` (PDG 3 -4 4 -4) is **missing entirely** from the check output.

### Root cause: the C++ check_sa.cpp flavor table was restructured by the merge
The test injects two extra flavors by string-patching `check_sa.cpp`
(test_cmd.py:1009-1026). It edits THREE things:
`static const int maxflavor`, `static const int flavor_arr[...]`, and
`static const int pdg_arr[...]`.

But the merged standalone_cpp `check_sa.cpp` no longer has a `flavor_arr`
array. The generated header now contains only:
```
static const int maxflavor  = 18;
static const int pdg_arr[18][4] = {{1,-1,1,-1}, ... 18 rows ...};
```
(There is no `static const int flavor_arr` line at all.) So the test's
`flavor_arr` patch branch (lines 1019-1021) silently matches nothing: only
`maxflavor` (->20) and `pdg_arr` (+2 rows) get edited. The internal
flavor->matrix-element mapping that used to be driven by `flavor_arr` is now
absent/changed, so the two injected rows evaluate to a wrong/zero result and
the second one never makes it into the parsed output.

The Fortran path (`check_sa.f`, MAXFLAVOR + FLAVOR/PDG_FOR_FLAVOR arrays,
lines 965-985) was NOT restructured, which is why only the C++ half fails.

### Proposed lines of action (NOT applied)
1. Inspect the new standalone_cpp `check_sa.cpp` flavor model to learn how
   flavor -> matrix-element is now driven (only `pdg_arr` present; understand
   what replaced `flavor_arr` and how `iflav` selects the channel — see the
   loop at `for(int iflav = 0; iflav < maxflavor; iflav++)`).
2. Rewrite the C++ injection block (test_cmd.py:1009-1026) to match the new
   single-array structure: bump `maxflavor`, extend `pdg_arr`, and supply
   whatever the new code uses in place of `flavor_arr` (possibly nothing, if
   flavor is now derived from `pdg_arr` directly).
3. Re-confirm the MASKDBG dump markers in `CPPProcess.cc` (lines 1028-1048:
   `ixxxxx(p[perm[0]]`, `current_wf_mask`, `nwords_wf`, etc.) still exist with
   the same names after the ALOHAOBJ refactor; if the mask variable names
   changed, `parse_mask`/the dump will also need updating.

### Note
This is a genuine merge impact on the test's white-box patching of generated
C++ (not a mode-default issue — the test uses explicit `output standalone` /
`output standalone_cpp`). The physics is likely fine; the test's source-surgery
is stale.

---

## 4. `test_madspin_wplus_all_all_flavor_balance` (tests/acceptance_tests/test_madspin.py:392)

### STATUS: RESOLVED
Applied the proposed fix: both this test and the `_2to1` sibling now use
`output madevent %(path)s`. Verified both pass (83s) and the flavour-balance
assertions hold cleanly once the madevent run path is restored -- e/mu/tau and
quark counts are balanced (e.g. -11:212, -13:213, -15:214; 2:677, 4:684). So the
merge introduced no hidden flavour-balance regression; the only issue was the
mg7 default output.

### Symptom
```
AssertionError: [] is not true
  at _get_single_decayed_lhe_path(), test_madspin.py:65  assertTrue(decayed_events)
```
No MadSpin-decayed LHE file is produced; the glob for decayed events is empty.
Test wall time ~0.8s (the launch/madspin pipeline never really ran).

### Root cause
Same bare-`output` -> mg7 default class as the other MadSpin failures. The
input script (test_madspin.py:401) does a bare `output %(path)s`, which since
`83c090d70` defaults to mg7. The subsequent `launch` + `madspin=ON` has no
madevent run directory to operate on, so no `*_decayed_*` LHE is generated and
the flavor-balance assertions never get a chance to run. (The earlier
`return_code == 0` check passes because the script "completes", just without
producing the madspin output.)

### Proposed action (NOT applied)
- Pin the script to madevent: test_madspin.py:401 `output %(path)s` ->
  `output madevent %(path)s` (same one-line fix used for the other MadSpin
  acceptance tests this session).
- THEN re-evaluate the actual subject of the test -- the e/mu/tau and
  quark flavour-balance assertions (lines 436-452). Those could not run while
  the output was mg7, so it is unknown whether the merge also affects the
  `decay w+ > all all` flavour balance. Recommend running once pinned to
  madevent to confirm the balance assertions pass before closing this out.
  (There is a sibling `test_madspin_wplus_all_all_flavor_balance_2to1` at
  test_madspin.py:454 that almost certainly needs the same pin.)
