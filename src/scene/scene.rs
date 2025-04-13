use super::SceneObject;
use crate::vector_traits::BaseVector;
use by_address::ByThinAddress;
use num_traits::{FromPrimitive, PrimInt};
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{Deref, Index};

#[derive(Clone, Debug)]
pub(crate) struct Scene<Vector>
where
    Vector: BaseVector,
{
    objects: Vec<SceneObject<Vector>>,
}

// fixme move index map somewhere else
/// A mapping datastructure that alloows accessing & looking up elements by a reference or an index
#[derive(Clone, Debug)]
struct IndexedMap<'a, I, V> {
    /// Values
    vs: &'a [V],
    /// Index to values map
    i2v: HashMap<I, &'a V>,
    /// Value-reference to index map
    v2i: HashMap<ByThinAddress<&'a V>, I>,
}

impl<'a, V> IndexedMap<'a, (), V> {
    #[inline(always)]
    pub fn from_slice<II>(vs: &'a [V]) -> IndexedMap<'a, II, V>
    where
        II: Eq + Hash + FromPrimitive + PrimInt,
    {
        vs.into()
    }
}

impl<'a, I, V> Deref for IndexedMap<'a, I, V> {
    type Target = HashMap<I, &'a V>;

    fn deref(&self) -> &Self::Target {
        &self.i2v
    }
}

impl<'a, I, V> Index<I> for IndexedMap<'a, I, V>
where
    I: Eq + Hash,
{
    type Output = &'a V;

    fn index(&self, index: I) -> &Self::Output {
        self.i2v.get(&index).expect("Index out of bounds")
    }
}

impl<'a, I, V> From<&'a [V]> for IndexedMap<'a, I, V>
where
    I: Hash + Eq + FromPrimitive + PrimInt,
{
    fn from(vs: &'a [V]) -> Self {
        assert!(vs.len() <= I::max_value().to_usize().unwrap());

        let (i2v, v2i) = vs
            .iter()
            .enumerate()
            .map(|(idx, v)| {
                let idx = I::from_usize(idx).unwrap();
                ((idx, v), (ByThinAddress(v), idx))
            })
            .collect::<(HashMap<_, _>, HashMap<_, _>)>();

        Self { vs, i2v, v2i }
    }
}

impl<'a, I, V> IndexedMap<'a, I, V>
where
    I: Copy,
{
    fn index_of(&self, v: &'a V) -> Option<I> {
        self.v2i.get(&ByThinAddress(v)).copied()
    }
}

/// Struct to hold read-only scene data references.
#[derive(Clone, Debug)]
pub(crate) struct SceneContext<'a, Vector>
where
    Vector: BaseVector,
{
    objects: IndexedMap<'a, u32, SceneObject<Vector>>,
}

impl<'a, Vector> From<&'a Scene<Vector>> for SceneContext<'a, Vector>
where
    Vector: BaseVector,
{
    fn from(scene: &'a Scene<Vector>) -> Self {
        assert!(!scene.objects.len() <= u32::MAX as usize);

        Self {
            objects: IndexedMap::from_slice(scene.objects.as_slice()),
        }
    }
}

impl<'a, Vector> SceneContext<'a, Vector>
where
    Vector: BaseVector,
{
    pub(crate) fn get_object_index(&self, object: &'a SceneObject<Vector>) -> Option<u32> {
        self.objects.index_of(object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod test_index_map {
        use super::*;
        #[test]
        fn test_insert_retrieve() {
            #[derive(Debug, PartialEq)]
            struct TestStruct(u8);

            let slice = &[TestStruct(1), TestStruct(2), TestStruct(3)];
            let map = IndexedMap::from_slice::<u8>(slice);

            assert_eq!(
                map.index_of(&TestStruct(1)),
                None,
                "Index of should only return Some when the SAME reference is passed in"
            );
            assert_eq!(map.index_of(&slice[1]), Some(1));

            assert_eq!(map[1], &slice[1]);
            assert_eq!(map[0], &TestStruct(1));
        }
    }
}
