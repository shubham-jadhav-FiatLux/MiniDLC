## Summary

<!-- What does this change, and why? Link the issue it addresses, if any. -->

## Testing

<!-- What did you run? Which new tests fail without the change? -->

- [ ] `ctest --test-dir build --output-on-failure` passes (Release)
- [ ] The ASan/UBSan build passes, for C++ changes
- [ ] Added or updated tests that fail without this change
- [ ] Updated the docs (`docs/passes/`, `docs/architecture.md`) if behavior changed
- [ ] Regenerated the GPU artifacts (`examples/gpu_artifacts.py`) if lowering or code generation changed

## Performance

<!-- For performance changes: before/after numbers, the script, and the machine. Otherwise delete this section. -->
