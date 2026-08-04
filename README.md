# CpliceAI

Tensorflow C API requires:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

## Testing

End-to-end tests for `cpliceai_reference`, `cpliceai_predict_variant`, and `cpliceai_predict_gene`
live under `tests/` as bats-core scripts, wired into CTest and a `check` target:

```
cmake -S . -B build
cmake --build build -j
cmake --build build --target check   # equivalent to: cd build && make check
```

`ctest --test-dir build -L cli` runs just the fast argument-parsing tests; `-L e2e` runs the full
pipeline test, which loads the real SpliceAI TensorFlow models against a small synthetic fixture
(`tests/fixtures/`) and takes tens of seconds.

Requires `bats` on `PATH` (installed in `.devcontainer/Dockerfile`
