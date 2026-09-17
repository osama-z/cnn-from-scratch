# Contributing

Start with the [setup guide](docs/setup.md) and [project layout](README.md#project-layout).
Use a focused branch and describe the behavior your pull request changes.

From the repository root:

```bash
make test-software
make docs-check
# Required for changes to RTL:
make test-hardware
# Required for changes to native memory handling or parsing:
make asan
```

Add a regression test when fixing a functional bug. For numerical changes,
explain the tensor layout, arithmetic assumptions, and comparison tolerances.
For performance claims, include the target, build configuration, workload, and
what is inside the timer. Clearly label simulation and physical measurements.

The frozen Lesson 7 model is a compatibility fixture. Regenerate it with
`make -C lessons/day7 model` and explain any intentional byte changes.
Keep datasets, generated executables, model bundles, credentials, and personal
application documents out of commits.

Report a bug with the command, expected behavior, actual output, environment,
and a small reproducer. Remove credentials and personal details before sharing.
