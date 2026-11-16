# English research presentation update

The existing [Google Slides research deck](https://docs.google.com/presentation/d/1X6HSrvL_L7XZjNDd2qv1a4vUSrzd37tSb2n5ljms0W0/edit) contains the original 14 PD and uniform-current slides followed by eight English wake slides.

| Slide | Content |
|---:|---|
| 15 | English translation of the wake and target-layout figure |
| 16 | 100 upstream scenarios and cylinder surface arrival criterion |
| 17 | Actual MuJoCo fish geometry renders and sensor inventory |
| 18 | Local wake drag coupling and six-input controller boundary |
| 19 | Recorded arrival rates for three amplitudes in two flow conditions |
| 20 | Straight-ahead recorded head trajectories at three amplitudes |
| 21 | Left, straight and right arrivals at the maximum amplitude |
| 22 | Scope of the prescribed wake model and reproduction files |

`make_figures_en.py` recreates the plots directly from the benchmark data and renders the actual included XML morphologies. `figures_en/` contains the English images. `slide-plan.json` records the source deck exemplars and media mapping. `research_vortex_update.pdf` is the verified final PDF export. Narrative slide text remains editable in Google Slides. Scientific plots remain images generated from the supplied reproducible scripts.

Source deck snapshots and temporary QA artifacts live in the local `research-update/` directory and are excluded from Git.
