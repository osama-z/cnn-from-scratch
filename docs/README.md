# Documentation

Start with the [main guide](guide.md) if the concepts are new, or
[setup](setup.md) if you want to run the code first.

## Read in this order

1. [Understand the CNN](guide.md): follow an image through the network, then learn
   how training and deployment fit together.
2. [Setup and commands](setup.md): dependencies, tests, training, and the demo.
3. [Lessons 1–7](../lessons/README.md): small examples in learning order.
4. [Verification](verification.md): what each test checks and what it cannot prove.
5. [Model release](release.md): run the pre-trained demo and rebuild the bundle.

## Implementation and references

| Topic | Location |
|---|---|
| Training and deployment scripts | [trained/](../trained/README.md) |
| Shared layers and binary reader/writer | [engine/](../engine/README.md) |
| VHDL blocks and simulation | [hardware/vhdl/](../hardware/vhdl/README.md) |
| Equations and tensor layouts | [Formula reference](reference/formulas.md) |
| Binary model specification | [Model format v1](reference/model-format.md) |
| Measured accuracy and CPU performance | [Baseline report](../reports/trained-baseline.md) |

## Project history

These files preserve the original learning journey. They contain historical
plans, not a statement that every planned feature exists today.

- [Learning log](history/learning-log.md)
- [Original roadmap](history/roadmap.md)

[Back to the project](../README.md)
