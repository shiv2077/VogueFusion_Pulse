# tests/color_science

Tests for the colour model the dye solver is built on
(`src/costing/color/perceptual.py`) and for the solver itself
(`src/costing/solver.py`). See `docs/DECISIONS.md` ADR-004 to ADR-008.

## Naming convention

- **`test_de2000_*`, `test_km_*`, and the general Lab and Kubelka-Munk
  tests** check `perceptual.py` against published references (Sharma, Wu and
  Dalal 2005 for CIEDE2000) and against `colour-science`, which is used only
  as a test oracle, never at runtime (ADR-006).
- **`test_rgb_*`** describe how the original approach (raw sRGB distance and
  sRGB-averaged mixing) fails. They pass: they assert facts about the old
  approach.
- **`test_dye_solver_contract.py`** started as an `xfail(strict=True)`
  specification run against a stub of the original solver. It now runs the
  real solver and every test passes. Three assertions changed shape when the
  stub was replaced; the reasons are at the top of that file.

## Running

```
pytest tests/color_science -q
```

Expected result: all pass (66 tests).
