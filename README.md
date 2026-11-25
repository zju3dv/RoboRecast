<h1 align="center">RoboRecast: Intent-Preserving Demonstration Augmentation via Diagnosis-Driven Optimization</h1>

<p align="center">Yushi Long<sup>1</sup>, Juzhan Xu<sup>1</sup>, Haibin Huang<sup>3</sup>, Chi Zhang<sup>3</sup>, Yin Chen<sup>3</sup>, Sida Peng<sup>1</sup>, Ruizhen Hu<sup>2,†</sup>, Xiaowei Zhou<sup>1</sup></p>

<p align="center">¹ Zhejiang University · ² Shenzhen University · ³ China Telecom<br>† Corresponding author</p>

<p align="center"><a href="https://zju3dv.github.io/RoboRecast/">Project Page</a> · <a href="https://zju3dv.github.io/RoboRecast/#demos">Demonstration Gallery</a> · <a href="https://zju3dv.github.io/RoboRecast/#video">Demo Video</a></p>

![Source human and robot demonstrations, diagnosis of failed interactions, and variations in layout, object, and robot.](docs/assets/teaser.webp)

Through diagnosis-driven optimization, RoboRecast augments real-world human and robot demonstrations for new layouts, objects, and robots while preserving task intent.

## Method Overview

RoboRecast identifies the contacts and state changes that define task intent in a real demonstration. The agent maps the roles in each interaction to entities in the target configuration, then uses these requirements to initialize the simulated scene and robot trajectory. If a validity check fails or a rollout misses a required interaction, the agent uses visual and numerical feedback to diagnose the cause, revise the relevant scene or motion parameters, and test the revision.

[View method overview →](https://zju3dv.github.io/RoboRecast/#method)

## Code Release Plan

- [ ] General-purpose skills and tools for the RoboRecast pipeline
- [ ] DROID example for Real2Sim reconstruction and demonstration augmentation
- [ ] EgoDex example for Human2Robot transfer
- [ ] Mobile ALOHA example for wheeled mobile manipulation

## Citation

```bibtex
@misc{long2026roborecast,
  title = {RoboRecast: Intent-Preserving Demonstration Augmentation via Diagnosis-Driven Optimization},
  author = {Long, Yushi and Xu, Juzhan and Huang, Haibin and Zhang, Chi and
            Chen, Yin and Peng, Sida and Hu, Ruizhen and Zhou, Xiaowei},
  year = {2026}
}
```
