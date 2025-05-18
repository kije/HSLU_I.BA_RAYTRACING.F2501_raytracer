mod lighting;
mod scene;

pub use lighting::{
    AmbientLight, Light, Lightable, PointLight, SceneLightSource, /*ShadingPipeline*/
};
pub use scene::Scene;
