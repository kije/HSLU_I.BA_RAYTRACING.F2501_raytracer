# Raytracer for the module "I.BA_RAYTRACING.F2501" at [HSLU](https://www.hslu.ch/en/)

## Prerequsites

- [Rust Nightly](https://www.rust-lang.org/tools/install)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -default-toolchain nightly
  ```

## Run

With SIMD

```bash
cargo +nightly run --profile release --features simd_render
```

Without SIMD

```bash
cargo +nightly run --profile release
```

### Semsterbild

```bash
cargo +nightly run --profile release --example semesterbild 
```

## Feature-Flags

The following feature flags are available that can be enabled/disabled via `--feature <feature>`

- simd_render = []
- anti_aliasing = []
- anti_aliasing_rotation_scale = ["anti_aliasing"]
- anti_aliasing_randomness = ["anti_aliasing"]
- high_resolution = []
- medium_resolution = []
- soft_shadows = []
- reflections = []
- refractions = []
- backface_culling = []
- scene_backface_culling = []
- save_rendering_image = []
- realistic = ["reflections", "light_reflections", "refractions"]
- high_quality_model = []
- high_quality = ["anti_aliasing", "soft_shadows", "high_quality_model"]
- extreme_quality = ["high_quality"]