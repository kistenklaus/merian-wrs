# Merian Example: Compute Sum

This is a minimal example demonstrating how to compute a sum of an array on the GPU using [Merian](https://github.com/LDAP/merian).

### Compile and run

```bash
meson setup build
# optionally for time measurements
meson configure build -Dmerian:performance_profiling=true
meson compile -C build

./build/merian-example
```


