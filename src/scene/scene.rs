use crate::geometry::GeometryCollection;
use crate::geometry::{SphereData, TriangleData};
use crate::simd_compat::SimdValueRealSimplified;
use crate::vector::Vector;
use std::collections::HashMap;

use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader, Read};

use crate::helpers::ColorType;
use crate::raytracing::{Material, TransmissionProperties};
use crate::{RENDER_RAY_FOCUS, SCENE_DEPTH, WINDOW_HEIGHT, WINDOW_WIDTH};
use itertools::Itertools;

use nobject_rs::Vertex;
use std::path::Path;
use ultraviolet::Vec3;
use wavefront_obj::ParseError;
use wavefront_obj::mtl::parse as parse_mtl;
use wavefront_obj::obj::parse as parse_obj;

#[derive(Debug, Clone)]
pub struct Scene<V: Vector<Scalar: SimdValueRealSimplified>> {
    pub scene_objects: GeometryCollection<V>,
}

impl<V: Vector<Scalar: SimdValueRealSimplified>> Scene<V> {
    pub fn new() -> Self {
        Self {
            scene_objects: GeometryCollection::new(),
        }
    }

    pub fn from_obj<P: AsRef<Path>>(path: P) -> Result<Scene<Vec3>, nobject_rs::ObjError> {
        let mut input_obj = BufReader::new(File::open(&path).unwrap());

        let content_input_obj = std::str::from_utf8(input_obj.fill_buf().unwrap()).unwrap();
        let parsed_obj = nobject_rs::load_obj(content_input_obj)?;

        let materials: HashMap<_, _> = parsed_obj
            .material_libs
            .iter()
            .filter_map(|lib_name| {
                let mut input_mtl =
                    BufReader::new(File::open(path.as_ref().parent()?.join(lib_name)).ok()?);
                let content_input_mtl = std::str::from_utf8(input_mtl.fill_buf().unwrap()).unwrap();
                let parsed_mtl = nobject_rs::load_mtl(content_input_mtl).unwrap();

                Some(parsed_mtl)
            })
            .flatten()
            .inspect(|m| println!("Loaded mtl: {:?}", m))
            .filter_map(|material| {
                //match material.illumination_mode.unwrap_or(0) {}

                Some((
                    material.name,
                    Material::new(
                        match material.diffuse {
                            Some(nobject_rs::ColorType::Rgb(r, g, b)) => ColorType::new(r, g, b),
                            _ => return None,
                        },
                        0.0,
                        material.specular_exponent.unwrap_or(0.0),
                        match material.transparancy {
                            Some(opacity) => TransmissionProperties::new(
                                opacity,
                                material.index_of_refraction.unwrap_or(1.0),
                            ),
                            _ => return None,
                        },
                    ),
                ))
            })
            .collect();

        println!("{parsed_obj:#?}");
        println!("{materials:#?}");
        // let obj: Obj<TexturedVertex> = Obj::new(parsed_obj)?;
        // //dome.material_libraries
        // println!("{obj:#?}");
        //
        let mut s = Scene::<Vec3>::new(); // capacity right?

        for (group_name, face) in parsed_obj.faces.iter() {
            let group = parsed_obj.groups.get(group_name);
            let group_material = group.and_then(|group| materials.get(&group.material_name));

            for face in face {
                let (vertices, normals): (Vec<_>, Vec<_>) = face
                    .elements
                    .iter()
                    .filter_map(|element| {
                        let normal = element
                            .normal_index
                            .and_then(|i| parsed_obj.normals.get(i as usize - 1));
                        let vertex = parsed_obj.vertices.get(element.vertex_index as usize - 1)?;

                        Some((vertex, normal))
                    })
                    .collect();

                let normals: Vec<_> = normals.into_iter().filter_map(|x| x).collect();

                if vertices.len() != 3 {
                    println!("Face with more than 3 vertices found. Skipping {vertices:#?}");
                    continue;
                };

                let (v1, v2, v3) = vertices
                    .iter()
                    .map(|v: &Vertex| Vec3::new(v.x, v.y, v.z))
                    .collect_tuple()
                    .unwrap();

                let face_material = group_material
                    .unwrap_or(&Material::new(
                        ColorType::new(1.0, 1.0, 1.0),
                        0.0,
                        0.0,
                        TransmissionProperties::none(),
                    ))
                    .clone();

                let triangle = if normals.len() >= 1 {
                    TriangleData::with_material_and_normal(
                        v1,
                        v2,
                        v3,
                        Vec3::new(normals[0].x, normals[0].y, normals[0].z),
                        face_material,
                    )
                } else {
                    TriangleData::with_material(v1, v2, v3, face_material)
                };

                s.add_triangle(triangle);
                // add normal for debug
                /*s.add_sphere(SphereData::new(
                    triangle.get_center() + 7.0 * triangle.normal,
                    3.0,
                    ColorType::new(15.0, 0., 0.),
                ));*/
            }
        }

        //println!("{s:#?} objects loaded");

        Ok(s)

        //Err(nobject_rs::ObjError::UnexpectedToken("FOO".to_string()))
    }

    pub fn with_capacities(scene_objects: usize) -> Self {
        Self {
            scene_objects: GeometryCollection::with_capacity(scene_objects),
        }
    }

    pub fn add_sphere(&mut self, sphere: SphereData<V>) {
        self.scene_objects.add(sphere.into());
    }

    pub fn add_triangle(&mut self, triangle: TriangleData<V>) {
        self.scene_objects.add(triangle.into());
    }
}
