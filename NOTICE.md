# EvalSim third-party notices and use restrictions

EvalSim's M3-and-later Waymo integration uses the Waymo Open Dataset and the
Waymax Licensed Materials. This repository is developed and distributed on the
operating assumption that it is used only for personal, non-commercial interview
preparation and personal experimentation.

This notice records upstream attribution and use conditions. It is not legal
advice, is not a substitute for the complete upstream agreements, and does not
represent that every use or distribution of EvalSim complies with those
agreements. The complete agreements control.

## Waymo Open Dataset

The work-type placeholder in the attribution prescribed by the March 2025 Waymo
Dataset License Agreement for Non-Commercial Use is instantiated below as
“software”:

> This software was made using the Waymo Open Dataset, provided by Waymo LLC
> under the Waymo Dataset License Agreement for Non-Commercial Use, available at
> waymo.com/open/terms, and your access and use of such work are governed by the
> terms and conditions therein.

Complete terms: <https://waymo.com/open/terms/>

No Waymo Open Dataset data, TFRecord shard, dataset extract, or generated
real-data output is included in this repository or its distributions. Dataset
access and use remain governed by the complete terms above. Among other
conditions, those terms limit use to Non-commercial Purposes and prohibit use or
deployment of the Dataset, applicable models, weights, or biases in operating or
assisting the operation of a vehicle, in Production Systems, or for other
primarily commercial purposes. This is only a non-exhaustive summary.

## Waymax

The work-type placeholder in the notice prescribed by the Waymax License
Agreement for Non-Commercial Use is instantiated below as “software.” The
upstream notice spells “Waymx” as shown; that spelling is preserved verbatim:

> This software was made using the Waymax Licensed Materials, provided by Waymo
> LLC under the Waymax License Agreement for Non-Commercial Use, available at
> https://github.com/waymo-research/waymax/blob/main/LICENSE, and your access and
> use of the Waymx Licensed Materials are governed by the terms and conditions
> contained therein.

```bibtex
@inproceedings{waymax, title={Waymax: An Accelerated, Data-Driven Simulator for
Large-Scale Autonomous Driving Research}, author={Cole Gulino and Justin Fu and
Wenjie Luo and George Tucker and Eli Bronstein and Yiren Lu and Jean Harb and
Xinlei Pan and Yan Wang and Xiangyu Chen and John D. Co-Reyes and Rishabh
Agarwal and Rebecca Roelofs and Yao Lu and Nico Montali and Paul Mougin and
Zoey Yang and Brandyn White and Aleksandra Faust, and Rowan McAllister and
Dragomir Anguelov and Benjamin Sapp}, booktitle={Proceedings of the Neural
Information Processing Systems Track on Datasets and Benchmarks},year={2023}}
```

Canonical license required by the prescribed notice:
<https://github.com/waymo-research/waymax/blob/main/LICENSE>

Immutable license provenance for the Waymax revision used by EvalSim:
<https://github.com/waymo-research/waymax/blob/a64dfec9be8576b60d9cecc94f406d9812d4a7d0/LICENSE>

No unmodified Waymax source code, Documentation, wheel, or cache is included in
this repository or its distributions. Waymax is obtained separately as an
optional dependency.

To the extent that EvalSim or a distribution of it is Derivative IP under the
Waymax License Agreement, access, use, modification, and conveyance are subject
to that agreement, and recipients must comply with all of its terms and
conditions. Among other conditions, the agreement:

- permits use only for Non-commercial Purposes;
- prohibits real-world vehicle operation or development of software or hardware
  for such operation;
- prohibits testing or validating the performance of any real-world vehicle;
- prohibits commercial driving-scenario simulation, use in Production Systems,
  and other primarily commercial purposes;
- prohibits conveying unmodified Waymax Licensed Materials; and
- prohibits using Waymax source code or Documentation, or derivatives of them,
  to train, develop, or improve an artificial-intelligence foundation model or
  a model distilled or fine-tuned from one.

This is only a non-exhaustive summary. Any license or agreement governing access
to applicable Waymax Derivative IP must require recipients to comply with the
complete Waymax License Agreement.

## EvalSim rights and purpose changes

Copyright in EvalSim remains with its respective owner or owners. This notice
does not itself license EvalSim and does not grant rights under the MIT, Apache,
or any other open-source or proprietary license. It also does not grant any
right to the Waymo Open Dataset or Waymax Licensed Materials beyond rights, if
any, granted directly by their respective upstream agreements.

If EvalSim's intended purpose changes from the personal, non-commercial use
described above—or may involve commercial use, a Production System, real-world
vehicle development, operation, testing or validation, or prohibited
foundation-model work—stop the affected Waymo Open Dataset and Waymax work and
obtain a fresh license review and any required permission before proceeding.
