use crate::geometry::{GeometryCollection, RenderGeometry};
use crate::geometry::{SphereData, TriangleData};
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::{Vector, VectorFixedLanes};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader, Read};

use crate::helpers::ColorType;
use crate::raytracing::{Material, TransmissionProperties};
use crate::{RENDER_RAY_FOCUS, SCENE_DEPTH, WINDOW_HEIGHT, WINDOW_WIDTH};
use itertools::Itertools;

use crate::float_ext::AbsDiffEq;
use crate::scene::lighting::SceneLightSource;
use crate::vector_traits::RenderingVector;
use std::path::Path;
use tobj::LoadError;
use ultraviolet::{Isometry3, Lerp, Mat4, Similarity3, Vec3};

#[derive(Debug, Clone)]
pub struct Scene<V: RenderingVector<Scalar: SimdValueRealSimplified>> {
    pub scene_objects: GeometryCollection<V>,
    pub scene_lights: Vec<SceneLightSource<V>>,
}

impl<V: RenderingVector<Scalar: SimdValueRealSimplified>> Scene<V> {
    pub fn new() -> Self {
        Self {
            scene_objects: GeometryCollection::new(),
            scene_lights: Vec::new(),
        }
    }

    pub fn merge(&mut self, other_scene: &Self) {
        self.scene_objects.merge(&other_scene.scene_objects);
        self.scene_lights.extend(&other_scene.scene_lights);
    }

    // fixme: Move mesh parsing code to a dedicated Mesh struct
    pub fn from_obj<P: AsRef<Path> + Debug, const CONTINUE_ON_MATERIAL_FAILURE: bool>(
        path: P,
        transform: Option<Similarity3>,
    ) -> Result<Scene<Vec3>, LoadError> {
        let transform_similarity = transform.unwrap_or_else(Similarity3::identity);
        let (models, materials) = tobj::load_obj(
            &path,
            &tobj::LoadOptions {
                triangulate: true,
                ignore_points: false,
                single_index: true,
                ..tobj::LoadOptions::default()
            },
        )?;

        let materials: Vec<Material<_>> = if CONTINUE_ON_MATERIAL_FAILURE {
            materials.unwrap_or(Vec::new())
        } else {
            materials?
        }
        .iter()
        .map(|material| material.into())
        .collect();

        let mut s = Scene::<Vec3>::with_capacities(models.len() * 10, 1); // fixme capacities
        for m in models.iter() {
            let mesh = &m.mesh;
            let material = mesh.material_id.and_then(|mid| materials.get(mid));

            mesh.indices
                .iter()
                .map(|idx| {
                    let i = *idx as usize;
                    let pos = transform_similarity.transform_vec(Vec3::new(
                        mesh.positions[3 * i],
                        mesh.positions[3 * i + 1],
                        mesh.positions[3 * i + 2],
                    ));
                    let normal = if !mesh.normals.is_empty() {
                        Some(
                            Vec3::new(
                                mesh.normals[3 * i],
                                mesh.normals[3 * i + 1],
                                mesh.normals[3 * i + 2],
                            )
                            .rotated_by(transform_similarity.rotation),
                        )
                    } else {
                        None
                    };

                    (i, pos, normal)
                })
                .chunks(3)
                .into_iter()
                .map(|data| {
                    let [v1, v2, v3] = data
                        .collect_array::<{ 3 }>()
                        .expect("Failed to collect array");
                    let n = match (v1.2, v2.2, v3.2) {
                        (None, None, None) => None,
                        (Some(n), None, None) | (None, Some(n), None) | (None, None, Some(n)) => {
                            Some(n)
                        }
                        (Some(n1), Some(n2), None)
                        | (Some(n1), None, Some(n2))
                        | (None, Some(n1), Some(n2)) => Some(n1.lerp(n2, 0.5)),
                        (Some(n1), Some(n2), Some(n3)) => Some(n1.lerp(n2, 0.5).lerp(n3, 0.5)),
                    };

                    let face_material = material
                        .cloned()
                        .unwrap_or(Material::diffuse(ColorType::new(1.0f32, 1.0, 1.0)));
                    if let Some(normal) = n {
                        TriangleData::with_material_and_normal(
                            v1.1,
                            v2.1,
                            v3.1,
                            normal,
                            face_material,
                        )
                    } else {
                        TriangleData::with_material(v1.1, v2.1, v3.1, face_material)
                    }
                })
                .for_each(|triangle| {
                    s.add_triangle(triangle);
                });
        }

        Ok(s)
    }

    pub fn backface_culling(scene: Scene<Vec3>, view_direction: Vec3) -> Scene<Vec3> {
        Scene {
            scene_lights: scene.scene_lights.clone(),
            scene_objects: scene
                .scene_objects
                .get_all()
                .cloned()
                .into_iter()
                .filter(|object| match object {
                    RenderGeometry::Sphere(_) => true,
                    RenderGeometry::Triangle(triangle) => triangle
                        .normal
                        .dot(view_direction)
                        .abs_diff_ne_default(&0.0),
                })
                .collect(),
        }
    }

    pub fn with_capacities(scene_objects: usize, scene_lights: usize) -> Self {
        Self {
            scene_objects: GeometryCollection::with_capacity(scene_objects),
            scene_lights: Vec::with_capacity(scene_lights),
        }
    }

    pub fn add_sphere(&mut self, sphere: SphereData<V>) {
        self.add_geometry(sphere.into());
    }

    pub fn add_triangle(&mut self, triangle: TriangleData<V>) {
        self.add_geometry(triangle.into());
    }

    pub fn add_geometry(&mut self, geometry: RenderGeometry<V>) {
        self.scene_objects.add(geometry);
    }

    pub fn add_light(&mut self, light: SceneLightSource<V>) {
        self.scene_lights.push(light)
    }
}
