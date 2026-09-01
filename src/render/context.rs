// src/render/context.rs

use crate::render::resource::{
    GraphResource, ImportedTexture, ResolvedResource, ResourceId, TransientResource, ViewId,
};

pub struct RenderContext {
    resources: Vec<Option<GraphResource>>,
}

impl RenderContext {
    /// Construct from pool-owned allocations only — the CPU-upload path, and what
    /// `CompiledGraph::execute` builds when a frame imports nothing.
    pub fn new(resources: Vec<Option<TransientResource>>) -> Self {
        Self {
            resources: resources
                .into_iter()
                .map(|slot| slot.map(GraphResource::Transient))
                .collect(),
        }
    }

    /// Construct from a mix of pool-owned and imported slots.
    ///
    /// G2b. Kept distinct from [`Self::new`] so the ownership of every slot is
    /// decided by the caller that acquired it, rather than inferred here: the
    /// release path at the end of a frame is driven entirely by which variant each
    /// slot holds (see [`Self::into_pooled`]).
    pub fn from_slots(resources: Vec<Option<GraphResource>>) -> Self {
        Self { resources }
    }

    /// Look up a resolved resource by its ResourceId.
    ///
    /// Imported and transient resources are indistinguishable here on purpose —
    /// a node samples what it was handed and never has to ask who owns it.
    ///
    /// # Panics
    /// Panics if `id` was not declared by any node's declare_resources(), or was
    /// declared as an import and the frame bound nothing to it.
    pub fn get(&self, id: ResourceId) -> ResolvedResource<'_> {
        self.try_get(id).unwrap_or_else(|| panic!("Resource {:?} not found in context", id))
    }

    /// Check if a resource was allocated (safe alternative to get()).
    pub fn contains(&self, id: ResourceId) -> bool {
        self.resources.get(id.0 as usize).is_some_and(|opt| opt.is_some())
    }

    /// Try to get a resolved resource
    pub fn try_get(&self, id: ResourceId) -> Option<ResolvedResource<'_>> {
        self.resources
            .get(id.0 as usize)
            .and_then(|opt| opt.as_ref())
            .map(GraphResource::resolved)
    }

    /// Whether `id` is bound to an externally-owned texture this frame.
    ///
    /// Exposed for tests and diagnostics, not for nodes: a node that branches on
    /// this is a node that will disagree with the graph about who frees what.
    pub fn is_imported(&self, id: ResourceId) -> bool {
        self.resources
            .get(id.0 as usize)
            .and_then(|opt| opt.as_ref())
            .is_some_and(GraphResource::is_imported)
    }

    /// The `ViewId` a slot is currently bound to, if any.
    pub fn view_id(&self, id: ResourceId) -> Option<ViewId> {
        self.resources
            .get(id.0 as usize)
            .and_then(|opt| opt.as_ref())
            .map(GraphResource::view_id)
    }

    /// Take the textures that must go back to the transient pool, and ONLY those.
    ///
    /// **This is the ownership boundary.** Imported slots are dropped here rather
    /// than yielded: releasing one would park a decoder-owned texture in a pool
    /// bucket, from which it would later be handed to an unrelated resource while
    /// the decoder still writes into it. That failure has no error and no counter —
    /// the pool would report an ordinary hit — so the filter lives in one place and
    /// `render::resource::tests::an_imported_resource_is_never_released_to_the_pool`
    /// pins it.
    pub fn into_pooled(self) -> impl Iterator<Item = TransientResource> {
        self.resources
            .into_iter()
            .flatten()
            .filter_map(GraphResource::into_transient)
    }

    /// Take ownership of every slot, both kinds, without deciding anything.
    pub fn into_resources(self) -> Vec<Option<GraphResource>> {
        self.resources
    }
}

/// Build a context's slot vector: `descriptors.len()` empty slots, then whatever
/// the caller fills in.
///
/// Free function rather than a method so the acquire loops in `CompiledGraph`'s
/// three `execute*` bodies stay identical to each other — the pool interaction is
/// the part that must not drift between them.
pub fn empty_slots(len: usize) -> Vec<Option<GraphResource>> {
    (0..len).map(|_| None).collect()
}

/// Bind an imported texture into a slot vector.
///
/// Panics if the slot vector is too short for `id`. It cannot be, for a graph the
/// compiler produced — `max_id` is taken over reads, and
/// [`crate::render::resource::ResourceBuilder::import`] records a read — so a short
/// vector means the caller built the slots by hand and the alternative is an import
/// that silently never binds.
pub fn bind_import(
    slots: &mut [Option<GraphResource>],
    id: ResourceId,
    texture: &ImportedTexture,
) {
    let len = slots.len();
    let slot = slots.get_mut(id.0 as usize).unwrap_or_else(|| {
        panic!("no slot for imported resource {id:?}: the context has {len} slot(s)")
    });
    *slot = Some(GraphResource::Imported(texture.clone()));
}
