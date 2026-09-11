"""Reconstruct the intermediate (pre-G2b) state of the two mixed files.

graph.rs and render_integration.rs each carry TWO uncommitted changes: the
node-timing/pool-stats plumbing that `src/bin/bench.rs` at HEAD already calls,
and the G2b imported-resource work. HEAD does not build without the first, so it
has to be committed first and separately. This writes the intermediate state:
HEAD + node timings, no imports.

CRLF is preserved by reading/writing bytes and splitting on the file's own
terminator.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

NODE_NAME_AND_POOL_STATS = """
    /// The name of the node at a registered index.
    ///
    /// Pairs with [`Self::execution_order`] so a caller can build the same
    /// index \u2192 name mapping [`Self::execute_timed`] returns, and assert the two
    /// agree \u2014 which is the only thing keeping a per-node timing report attached to
    /// the right shader.
    pub fn node_name(&self, node_idx: usize) -> &str {
        self.nodes[node_idx].name()
    }

    /// What the transient texture pool has done since this graph was compiled.
    ///
    /// Exposed so a caller can report allocation behaviour it would otherwise have
    /// to guess at. The pool's cap has to cover a whole frame's simultaneous
    /// resources \u2014 `execute` acquires them all before the first node and releases
    /// them all after the last \u2014 so a non-zero `evicted` means every frame is
    /// dropping textures it will immediately re-allocate. At 4K that is 66 MB per
    /// texture per frame. See [`crate::render::resource::POOL_BUCKET_CAPACITY`].
    pub fn pool_stats(&self) -> crate::render::resource::PoolStats {
        self.texture_pool.lock().unwrap().stats()
    }
"""

EXECUTE_TIMED = """
    /// Execute with one GPU timestamp bracket per node.
    ///
    /// Separate from [`Self::execute`] rather than a flag on it: the timing path
    /// needs a query set sized to the node count and a resolve recorded after the
    /// last node, and threading an `Option<&mut GpuTimer>` through the hot path
    /// would put a branch in every frame of the UI's preview loop for a
    /// benchmark's benefit.
    ///
    /// Returns the node names in execution order, aligned one-to-one with the
    /// timer's brackets, so the caller maps `pair[i]` \u2192 `names[i]` explicitly
    /// instead of re-deriving `self.order` and hoping the two agree.
    ///
    /// **The caller must size the timer for `execution_order().len()` brackets and
    /// call [`crate::render::gpu_timer::GpuTimer::reset`] first.** When the timer
    /// is too small, `GpuTimer::mark` silently stops writing at capacity \u2014 so this
    /// returns names for the nodes it *bracketed*, truncated to what the timer
    /// could hold, rather than a full list that would misalign every reading after
    /// the cut.
    ///
    /// **Timestamps go BETWEEN passes, not inside them.** Writing inside a pass
    /// needs `TIMESTAMP_QUERY_INSIDE_PASSES`, which nothing in this crate
    /// requests; every node opens and closes its own compute pass, so a bracket
    /// around `record` already gives per-node resolution.
    pub fn execute_timed<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        device: &GpuDevice,
        frame: &FrameState,
        timer: &mut crate::render::gpu_timer::GpuTimer,
    ) -> Vec<&'a str> {
        // Acquire exactly as `execute` does. Duplicated rather than shared because
        // the alternative is a closure per node in the hot path \u2014 and the pool
        // interaction is the part that must stay identical, which the pool's own
        // counters make checkable: a timed run and an untimed one must report the
        // same hit/miss/evicted numbers.
        let mut pool = self.texture_pool.lock().unwrap();
        let mut resources: Vec<Option<(wgpu::Texture, wgpu::TextureView, crate::render::resource::ViewId)>> =
            (0..self.descriptors.len()).map(|_| None).collect();

        for (id_usize, desc_opt) in self.descriptors.iter().enumerate() {
            if let Some((desc, usage)) = desc_opt {
                let (w, h) = match desc.size {
                    ResolutionSource::Canvas => (self.canvas_width, self.canvas_height),
                    ResolutionSource::Fixed(fw, fh) => (fw, fh),
                };
                resources[id_usize] = Some(pool.acquire(device, desc.format, *usage, w, h));
            }
        }
        drop(pool);

        let ctx = RenderContext::new(resources);

        let mut names: Vec<&'a str> = Vec::with_capacity(self.order.len());
        for &node_idx in &self.order {
            let node = &self.nodes[node_idx];
            // Only claim a name for a node whose bracket actually fits. `written`
            // stops advancing at capacity, so comparing before and after is how a
            // dropped bracket is detected \u2014 without it, an undersized timer would
            // shift every subsequent reading onto the wrong node and the report
            // would attribute one shader's cost to another.
            let before = timer.written();
            timer.begin(encoder);
            encoder.push_debug_group(node.name());
            node.record(encoder, &ctx, frame);
            encoder.pop_debug_group();
            timer.end(encoder);
            if timer.written() == before + 2 {
                names.push(node.name());
            }
        }
        timer.record_resolve(encoder);

        let mut pool = self.texture_pool.lock().unwrap();
        for res in ctx.into_resources().into_iter().flatten() {
            pool.release(res);
        }
        names
    }
"""


def rewrite(path, edits):
    raw = path.read_bytes()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for anchor, insert, where in edits:
        assert text.count(anchor) == 1, (path.name, anchor[:60], text.count(anchor))
        body = insert.replace("\r\n", "\n")
        text = (
            text.replace(anchor, anchor + body)
            if where == "after"
            else text.replace(anchor, body + anchor)
        )
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))
    print(f"rewrote {path.relative_to(ROOT)}")


graph = ROOT / "src/render/graph.rs"
rewrite(
    graph,
    [
        (
            "    /// Returns the compiled execution order of nodes by their registered indices.\n"
            "    pub fn execution_order(&self) -> &[usize] {\n"
            "        &self.order\n"
            "    }\n",
            NODE_NAME_AND_POOL_STATS,
            "after",
        ),
        ("    pub fn execute_with_callback<F>(", EXECUTE_TIMED + "\n", "before"),
    ],
)

# render_integration.rs: keep the execute_timed test, drop the G2b tests.
ri = ROOT / "src/tests/render_integration.rs"
work = (ROOT / ".hermes/g2b/render_integration.rs").read_bytes().decode("utf-8")
work = work.replace("\r\n", "\n")
marker = "\n    // \u2500\u2500 G2b \u2014 an imported resource, driven through a real graph \u2500\u2500\u2500"
assert work.count(marker) == 1, "G2b marker not found in the working copy"
intermediate = work.split(marker)[0].rstrip() + "\n}\n"
ri.write_bytes(intermediate.replace("\n", "\r\n").encode("utf-8"))
print(f"rewrote {ri.relative_to(ROOT)} ({intermediate.count(chr(10))} lines)")
