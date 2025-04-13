mod lighting;
mod scene;
mod scene_object;

pub(crate) use lighting::{AmbientLight, Light, Lightable, PointLight /*ShadingPipeline*/};
pub(crate) use scene_object::SceneObject;
