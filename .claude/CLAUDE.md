# norfair-rust Development Guide

## Debugging python→Rust Equivalence Issues

### THE GOLDEN RULE

**When tests fail showing numerical differences between python/go and Rust:**

**DO THIS FIRST (5 minutes):**
1. Find the Rust code that is failing
2. Find the correspinding python and go code
3. Open them all side-by-side
4. Compare them all line-by-line
5. Look for obvious bugs:
   - Wrong formulas (det(c*A) vs c^n*det(A))
   - Scalar vs array parameters
   - Missing loops or wrong loop bounds
   - Transposed matrices or wrong indexing
   - Incorrect logic or control flow
6. Add a test case for Rust that checks this divergence point.

**ONLY IF THAT FAILS (rare):**
1. Create minimal debug fixture in python
2. Add targeted debug output to Rust
3. Compare intermediate values
4. Trace divergence point
5. Add a test case for Rust that checks this divergence point.

### Basic Debugging Checklist

When Rust output ≠ pyhon output:

- [ ] **Did I port the algorithm correctly?** (Read both implementations side-by-side)
- [ ] **Am I using the right parameters?** (scalar vs array, indexing issues, broadcasting issues, etc)
- [ ] **Are my formulas correct?** (Check mathematical properties like determinant rules)
- [ ] **Did I test with a minimal example?** (Simplest possible input that fails)

## Development Reminders

- **Run tests with --release**: `cargo test --release`
- **Question assumptions**: If tests require loose tolerances (>1e-10), there's likely a real bug
- **Push back** if you notice weird hacks to achieve python equivalence
- **Keep PLAN.md updated** with current work status and next steps. Remember to update it after making logical progress.
- **Keep PLAN_TESTS.md updated** in the same way as PLAN.md
- **All tests** All tests must be ported.
- all python libs and commands need to be run through uv, e.g. GOOD: `uv run maturin` BAD: `pip install maturin ; maturin`