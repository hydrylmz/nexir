use crate::history::HistoryState;
use crate::layout::media_pool::{MediaEntry, MediaKind};
use egui::{Align2, Color32, Rect, RichText, Sense, Ui, Vec2};
use nexir::project::Project;
use crate::image_still::StillImageCache;
use nexir::timeline::ids::{ClipId, TrackId};
use nexir::timeline::mutation::{ClipInsertParams, remove_clip};
use nexir::timeline::track::TrackKind;
use nexir::timeline::transform::ClipTransform;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ResizeEdge {
    Left,
    Right,
}

enum ClipAction {
    Delete(ClipId),
    RippleDelete(ClipId),
    Duplicate(ClipId),
    SplitAtPlayhead(ClipId),
}

struct ClipResize {
    clip_id: ClipId,
    orig_idx: usize,
    edge: ResizeEdge,
    orig_pts_in: i64,
    orig_pts_out: i64,
}

/// State for an in-progress clip drag.
struct ClipDrag {
    /// Stable clip id being dragged.
    clip_id: ClipId,
    /// Store index at drag-start (used for appearance only during drag).
    orig_idx: usize,
    /// Pixel offset from the clip's left edge to where the user grabbed it.
    grab_offset_px: f32,
    /// Original track the clip lives on.
    orig_track: TrackId,
}

#[derive(Clone, Copy)]
pub struct ViewportResize {
    pub clip_id: ClipId,
    pub handle_index: usize,
    pub start_pointer: egui::Pos2,
    pub start_transform: ClipTransform,
}

#[derive(Clone, Copy)]
pub struct ViewportMove {
    pub clip_id: ClipId,
    pub start_pointer: egui::Pos2,
    pub start_transform: ClipTransform,
}

/// Timeline UI state owned by `NexirApp`.
pub struct TimelineState {
    pub playhead_frame: i64,
    pub scroll_x: f32,
    pub zoom: f32, // pixels per frame
    pub playing: bool,
    pub last_tick: Option<std::time::Instant>,
    pub selected_clip: Option<usize>,
    pub viewport_snap: bool,
    /// Active clip drag state.
    drag: Option<ClipDrag>,
    /// Active clip resize state.
    resize: Option<ClipResize>,
    viewport_resize: Option<ViewportResize>,
    viewport_move: Option<ViewportMove>,
    viewport_rotate: Option<ViewportRotate>,
}

#[derive(Clone, Copy)]
pub struct ViewportRotate {
    pub clip_id: ClipId,
    pub start_pointer: egui::Pos2,
    pub start_transform: ClipTransform,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            playhead_frame: 0,
            scroll_x: 0.0,
            zoom: 8.0,
            playing: false,
            last_tick: None,
            selected_clip: None,
            viewport_snap: true,
            drag: None,
            resize: None,
            viewport_resize: None,
            viewport_move: None,
            viewport_rotate: None,
        }
    }
}

impl TimelineState {
    pub fn is_interacting_with_clip(&self) -> bool {
        self.drag.is_some()
            || self.resize.is_some()
            || self.viewport_resize.is_some()
            || self.viewport_move.is_some()
            || self.viewport_rotate.is_some()
    }

    pub fn start_viewport_resize(
        &mut self,
        clip_id: ClipId,
        handle_index: usize,
        start_pointer: egui::Pos2,
        start_transform: ClipTransform,
    ) {
        self.viewport_resize = Some(ViewportResize {
            clip_id,
            handle_index,
            start_pointer,
            start_transform,
        });
    }

    pub fn viewport_resize(&self) -> Option<ViewportResize> {
        self.viewport_resize
    }

    pub fn finish_viewport_resize(&mut self) {
        self.viewport_resize = None;
    }

    pub fn start_viewport_move(
        &mut self,
        clip_id: ClipId,
        start_pointer: egui::Pos2,
        start_transform: ClipTransform,
    ) {
        self.viewport_move = Some(ViewportMove {
            clip_id,
            start_pointer,
            start_transform,
        });
    }

    pub fn viewport_move(&self) -> Option<ViewportMove> {
        self.viewport_move
    }

    pub fn finish_viewport_move(&mut self) {
        self.viewport_move = None;
    }

    pub fn start_viewport_rotate(
        &mut self,
        clip_id: ClipId,
        start_pointer: egui::Pos2,
        start_transform: ClipTransform,
    ) {
        self.viewport_rotate = Some(ViewportRotate {
            clip_id,
            start_pointer,
            start_transform,
        });
    }

    pub fn viewport_rotate(&self) -> Option<ViewportRotate> {
        self.viewport_rotate
    }

    pub fn finish_viewport_rotate(&mut self) {
        self.viewport_rotate = None;
    }

    pub fn clear_interaction(&mut self) {
        self.drag = None;
        self.resize = None;
        self.viewport_resize = None;
        self.viewport_move = None;
        self.viewport_rotate = None;
    }
}

pub fn draw(
    ui: &mut Ui,
    project: &mut Project,
    state: &mut TimelineState,
    dragging_item: &mut Option<MediaEntry>,
    history: &mut HistoryState,
    waveform_cache: &crate::waveform::WaveformCache,
    still_cache: &std::sync::Mutex<StillImageCache>,
) {
    let mut pending_clip_actions: Vec<ClipAction> = Vec::new();
    // ── Space to toggle play/pause ────────────────────────────────────
    if ui.input(|i| i.key_pressed(egui::Key::Space)) {
        state.playing = !state.playing;
        state.last_tick = if state.playing {
            Some(std::time::Instant::now())
        } else {
            None
        };
    }

    // ── Delete selected clip on Delete key (Shift+Delete for ripple) ──
    if ui.input(|i| i.key_pressed(egui::Key::Delete)) {
        if let Some(idx) = state.selected_clip {
            let clip_id = project.clips.clip_id_at(idx);
            let track_id = project.clips.track_id_at(idx);
            let is_audio = project.tracks.get(track_id).map_or(false, |t| {
                matches!(t.kind, nexir::timeline::track::TrackKind::Audio { .. })
            });

            let linked_id = if is_audio {
                None
            } else {
                find_linked_clip(project, clip_id)
            };
            history.record(project);
            let is_shift = ui.input(|i| i.modifiers.shift);
            if is_shift {
                let _ = project.ripple_remove_clip(clip_id);
                if let Some(l_id) = linked_id {
                    let _ = project.ripple_remove_clip(l_id);
                }
            } else {
                let _ = remove_clip(&mut project.clips, clip_id);
                if let Some(l_id) = linked_id {
                    let _ = remove_clip(&mut project.clips, l_id);
                }
            }
            state.selected_clip = None;
        }
    }

    // ── Ctrl+K to split selected clip at playhead ────────────────────
    let ctrl_k = ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::K));
    if ctrl_k {
        if let Some(idx) = state.selected_clip {
            let clip_id = project.clips.clip_id_at(idx);
            let pts_in = project.clips.pts_in_at(idx);
            let pts_out = project.clips.pts_out_at(idx);
            let playhead_pts = project.frame_to_pts(state.playhead_frame);
            if playhead_pts > pts_in && playhead_pts < pts_out {
                let linked_id = find_linked_clip(project, clip_id);
                history.record(project);
                let _ = nexir::timeline::mutation::split_clip(
                    &mut project.clips,
                    clip_id,
                    playhead_pts,
                );
                if let Some(l_id) = linked_id {
                    let _ = nexir::timeline::mutation::split_clip(
                        &mut project.clips,
                        l_id,
                        playhead_pts,
                    );
                }
                state.selected_clip = None;
            }
        }
    }

    // ── Pending track property mutations (collected while iterating immutably) ──
    enum TrackMutation {
        ToggleMute(TrackId),
        ToggleSolo(TrackId),
        AddVideoTrack,
        AddAudioTrack,
    }
    let mut pending_track_mutations: Vec<TrackMutation> = Vec::new();

    // ── Header / toolbar ─────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.strong(RichText::new("Timeline").color(Color32::WHITE));
        ui.separator();

        if ui.button("⏮").on_hover_text("Go to start").clicked() {
            state.playhead_frame = 0;
            state.playing = false;
        }
        if ui.button("◀◀").on_hover_text("Step back").clicked() {
            state.playing = false;
            state.playhead_frame = (state.playhead_frame - 1).max(0);
        }
        let play_label = if state.playing { "⏸" } else { "▶" };
        let play_hint = if state.playing { "Pause" } else { "Play" };
        if ui.button(play_label).on_hover_text(play_hint).clicked() {
            state.playing = !state.playing;
            state.last_tick = if state.playing {
                Some(std::time::Instant::now())
            } else {
                None
            };
        }
        if ui.button("▶▶").on_hover_text("Step forward").clicked() {
            state.playing = false;
            state.playhead_frame += 1;
        }
        if ui.button("⏭").on_hover_text("Go to end").clicked() {
            state.playing = false;
            state.playhead_frame = project.duration_frames();
        }

        ui.separator();



        // Timecode display — frame → HH:MM:SS:FF
        let fps = project.settings.frame_rate.num.max(1);
        let f = state.playhead_frame;
        let frames = f % fps;
        let secs = (f / fps) % 60;
        let mins = (f / (fps * 60)) % 60;
        let hours = f / (fps * 3600);
        ui.label(
            RichText::new(format!(
                "{:02}:{:02}:{:02}:{:02}",
                hours, mins, secs, frames
            ))
            .monospace()
            .color(Color32::LIGHT_GRAY),
        );

        ui.separator();
        let has_selection = state.selected_clip.is_some();
        if ui
            .add_enabled(has_selection, egui::Button::new("🗑 Delete"))
            .on_hover_text("Delete selected clip (Delete)")
            .clicked()
        {
            if let Some(idx) = state.selected_clip {
                let clip_id = project.clips.clip_id_at(idx);
                pending_clip_actions.push(ClipAction::Delete(clip_id));
            }
        }
        if ui
            .add_enabled(has_selection, egui::Button::new("🌊 Ripple Delete"))
            .on_hover_text("Ripple delete selected clip and close gap (Shift+Delete)")
            .clicked()
        {
            if let Some(idx) = state.selected_clip {
                let clip_id = project.clips.clip_id_at(idx);
                pending_clip_actions.push(ClipAction::RippleDelete(clip_id));
            }
        }

        ui.separator();
        let mut snap = state.viewport_snap;
        if ui
            .add(egui::SelectableLabel::new(snap, "Snap"))
            .on_hover_text("Snap clip to center in preview")
            .clicked()
        {
            state.viewport_snap = !snap;
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add(
                egui::Slider::new(&mut state.zoom, 0.1..=500.0)
                    .text("Zoom")
                    .clamp_to_range(true),
            );
        });
    });

    ui.separator();

    const GUTTER_W: f32 = 110.0;
    const RULER_H: f32 = 20.0;
    const CLIP_H_PAD: f32 = 4.0;

    let any_soloed = project.tracks.any_soloed();
    let total_frames = project.duration_frames().max(300);
    let fps = project.settings.frame_rate.num.max(1);

    // ── Collect track rects for ghost rendering (we need them during draw) ──
    // We'll build this during the track loop and use it for ghost + drop hints.
    let mut track_lane_rects: Vec<(TrackId, Rect)> = Vec::new();

    // ── Pending moves/drops applied after the draw loop ──────────────────
    struct PendingClipResize {
        clip_id: ClipId,
        edge: ResizeEdge,
        new_pts: i64,
    }
    let mut pending_clip_resize: Option<PendingClipResize> = None;

    struct PendingClipMove {
        clip_id: ClipId,
        new_track: TrackId,
        new_frame: i64,
    }
    let mut pending_clip_move: Option<PendingClipMove> = None;

    #[derive(Default)]
    struct PendingDrop {
        track_id: Option<TrackId>,
        drop_frame: i64,
        media_path: Option<std::path::PathBuf>,
    }
    let mut pending_drop = PendingDrop::default();

    // ── Is a clip currently being dragged? ───────────────────────────────
    let pointer_released = ui.input(|i| i.pointer.any_released());
    let pointer_pos = ui.ctx().pointer_hover_pos();

    // Helper: determine if a file is audio-only by extension
    let is_audio_file = |path: &std::path::Path| -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| {
                matches!(
                    e.to_lowercase().as_str(),
                    "mp3" | "wav" | "aac" | "ogg" | "flac" | "m4a" | "opus" | "wma"
                )
            })
            .unwrap_or(false)
    };

    // ── Timeline Zooming ─────────────────────────────────────────────────
    let pointer_in_timeline = ui.rect_contains_pointer(ui.max_rect());
    if pointer_in_timeline {
        let zoom_factor = ui.input(|i| {
            let mut factor = i.zoom_delta();
            let ctrl = i.modifiers.ctrl || i.modifiers.command;
            if ctrl {
                if i.raw_scroll_delta.y > 0.0 {
                    factor *= 1.2;
                } else if i.raw_scroll_delta.y < 0.0 {
                    factor *= 0.8333;
                }
            }
            factor
        });

        if zoom_factor != 1.0 {
            state.zoom = (state.zoom * zoom_factor).clamp(0.1, 500.0);
        }
    }

    // ── Ruler + lanes canvas ─────────────────────────────────────────────
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let canvas_w = total_frames as f32 * state.zoom + 60.0;

            // ── Ruler — click or drag to move playhead ─────────────
            let (ruler_rect, ruler_resp) = ui.allocate_exact_size(
                Vec2::new(canvas_w + GUTTER_W, RULER_H),
                Sense::click_and_drag(),
            );
            ui.painter()
                .rect_filled(ruler_rect, 0.0, Color32::from_rgb(20, 20, 20));

            // Move playhead on click/drag over ruler — but only when not dragging a clip
            if state.drag.is_none() && (ruler_resp.dragged() || ruler_resp.clicked()) {
                if let Some(pos) = ui.ctx().pointer_interact_pos() {
                    let raw_x = pos.x - ruler_rect.min.x - GUTTER_W;
                    let frame = (raw_x / state.zoom).max(0.0) as i64;
                    state.playhead_frame = frame.min(total_frames);
                    state.playing = false;
                }
            }

            // Draw tick marks
            let tick_interval: i64 = fps;
            let end_frame = total_frames + tick_interval;
            let mut f_mark = 0_i64;
            while f_mark <= end_frame {
                let x = GUTTER_W + f_mark as f32 * state.zoom;
                let tick_top = ruler_rect.min + egui::vec2(x, 0.0);
                let tick_bot = tick_top + egui::vec2(0.0, RULER_H * 0.5);
                ui.painter().line_segment(
                    [tick_top, tick_bot],
                    egui::Stroke::new(1.0, Color32::from_rgb(80, 80, 80)),
                );
                if f_mark % (tick_interval * 5) == 0 {
                    let secs = f_mark / fps;
                    ui.painter().text(
                        tick_top + egui::vec2(3.0, 2.0),
                        Align2::LEFT_TOP,
                        format!("{}s", secs),
                        egui::FontId::proportional(10.0),
                        Color32::from_rgb(120, 120, 120),
                    );
                }
                f_mark += tick_interval;
            }

            // ── Track rows ─────────────────────────────────────────
            // Sort tracks for display: Video on top, then Audio, etc.
            let mut sorted_tracks: Vec<&nexir::timeline::track::Track> = project.tracks.iter().collect();
            sorted_tracks.sort_by(|a, b| {
                fn kind_order(kind: &TrackKind) -> u8 {
                    match kind {
                        TrackKind::Video => 0,
                        TrackKind::Audio { .. } => 1,
                        TrackKind::Text => 2,
                        TrackKind::Effect => 3,
                    }
                }
                kind_order(&a.kind).cmp(&kind_order(&b.kind))
            });

            for current_kind_order in 0..=3 {
                let tracks_of_kind: Vec<_> = sorted_tracks.iter().filter(|t| {
                    (match t.kind {
                        TrackKind::Video => 0,
                        TrackKind::Audio { .. } => 1,
                        TrackKind::Text => 2,
                        TrackKind::Effect => 3,
                    }) == current_kind_order
                }).collect();

                if tracks_of_kind.is_empty() && current_kind_order > 1 {
                    continue;
                }

                if current_kind_order > 0 {
                    let gap_h = 12.0;
                    let (gap_rect, _) = ui.allocate_exact_size(Vec2::new(canvas_w + GUTTER_W, gap_h), Sense::hover());
                    let mid_y = gap_rect.center().y;
                    
                    // subtle separator spanning the entire width
                    ui.painter().line_segment(
                        [egui::pos2(gap_rect.left(), mid_y), egui::pos2(gap_rect.right(), mid_y)],
                        egui::Stroke::new(2.0, Color32::from_rgb(20, 20, 20)),
                    );
                }

                for &&track in tracks_of_kind.iter() {
                let is_active = track.is_active(any_soloed);
                let track_h = track.height_px as f32;

                let (row_rect, _) =
                    ui.allocate_exact_size(Vec2::new(canvas_w + GUTTER_W, track_h), Sense::hover());

                // Track gutter (left label area)
                let gutter = Rect::from_min_size(row_rect.min, Vec2::new(GUTTER_W, track_h));
                ui.painter()
                    .rect_filled(gutter, 0.0, Color32::from_rgb(28, 28, 28));

                // Track kind indicator stripe
                let stripe_color = match track.kind {
                    TrackKind::Video => Color32::from_rgb(40, 100, 190),
                    TrackKind::Audio { .. } => Color32::from_rgb(40, 160, 90),
                    TrackKind::Text => Color32::from_rgb(190, 120, 40),
                    TrackKind::Effect => Color32::from_rgb(140, 50, 190),
                };
                let stripe = Rect::from_min_size(gutter.min, Vec2::new(3.0, track_h));
                ui.painter().rect_filled(stripe, 0.0, stripe_color);

                // Track name
                let label_color = if is_active {
                    Color32::WHITE
                } else {
                    Color32::DARK_GRAY
                };
                ui.painter().text(
                    gutter.min + egui::vec2(10.0, track_h * 0.5),
                    Align2::LEFT_CENTER,
                    &track.name,
                    egui::FontId::proportional(12.0),
                    label_color,
                );

                // Mute / Solo buttons — interactive
                let m_rect = Rect::from_min_size(
                    gutter.min + egui::vec2(GUTTER_W - 46.0, (track_h - 16.0) * 0.5),
                    Vec2::splat(16.0),
                );
                let m_resp = ui.interact(m_rect, egui::Id::new(("mute", track.id)), Sense::click());
                let m_color = if track.mute {
                    Color32::YELLOW
                } else if m_resp.hovered() {
                    Color32::from_rgb(100, 100, 50)
                } else {
                    Color32::from_rgb(60, 60, 60)
                };
                ui.painter().rect_filled(m_rect, 3.0, m_color);
                ui.painter().text(
                    m_rect.center(),
                    Align2::CENTER_CENTER,
                    "M",
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
                );
                if m_resp.clicked() {
                    pending_track_mutations.push(TrackMutation::ToggleMute(track.id));
                }
                m_resp.on_hover_text(if track.mute {
                    "Unmute track"
                } else {
                    "Mute track"
                });

                let s_rect = Rect::from_min_size(
                    gutter.min + egui::vec2(GUTTER_W - 26.0, (track_h - 16.0) * 0.5),
                    Vec2::splat(16.0),
                );
                let s_resp = ui.interact(s_rect, egui::Id::new(("solo", track.id)), Sense::click());
                let s_color = if track.solo {
                    Color32::from_rgb(255, 160, 0)
                } else if s_resp.hovered() {
                    Color32::from_rgb(100, 70, 20)
                } else {
                    Color32::from_rgb(60, 60, 60)
                };
                ui.painter().rect_filled(s_rect, 3.0, s_color);
                ui.painter().text(
                    s_rect.center(),
                    Align2::CENTER_CENTER,
                    "S",
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
                );
                if s_resp.clicked() {
                    pending_track_mutations.push(TrackMutation::ToggleSolo(track.id));
                }
                s_resp.on_hover_text(if track.solo {
                    "Unsolo track"
                } else {
                    "Solo track"
                });

                // Lane background
                let lane = Rect::from_min_size(
                    row_rect.min + egui::vec2(GUTTER_W, 0.0),
                    Vec2::new(canvas_w, track_h),
                );
                track_lane_rects.push((track.id, lane));

                let is_effect_drag = dragging_item
                    .as_ref()
                    .map_or(false, |item| item.kind == MediaKind::Effect);

                // ── Determine lane highlight ──────────────────────────────
                // Priority: media-pool drag > clip drag > normal
                let is_media_drop_target = !is_effect_drag
                    && dragging_item.is_some()
                    && pointer_pos.map(|p| lane.contains(p)).unwrap_or(false);
                let is_clip_drag_target =
                    state.drag.is_some() && pointer_pos.map(|p| lane.contains(p)).unwrap_or(false);

                let is_compatible_hover = dragging_item
                    .as_ref()
                    .map(|item| {
                        let audio = is_audio_file(&item.path);
                        match track.kind {
                            TrackKind::Audio { .. } => audio,
                            TrackKind::Video => !audio,
                            _ => !audio,
                        }
                    })
                    .unwrap_or(true);

                let lane_bg = if is_media_drop_target && !is_compatible_hover {
                    Color32::from_rgb(75, 20, 20)
                } else if is_media_drop_target {
                    Color32::from_rgb(30, 50, 75)
                } else if is_clip_drag_target {
                    Color32::from_rgb(30, 55, 50) // teal tint for clip drag
                } else {
                    Color32::from_rgb(22, 22, 22)
                };
                ui.painter().rect_filled(lane, 0.0, lane_bg);

                // Subtle lane separator
                ui.painter().line_segment(
                    [lane.left_bottom(), lane.right_bottom()],
                    egui::Stroke::new(1.0, Color32::from_rgb(35, 35, 35)),
                );

                // ── Handle media-pool file drop ───────────────────────────
                if is_media_drop_target {
                    let is_compatible = dragging_item
                        .as_ref()
                        .map(|item| {
                            let audio = is_audio_file(&item.path);
                            match track.kind {
                                TrackKind::Audio { .. } => audio,
                                TrackKind::Video => !audio,
                                _ => !audio,
                            }
                        })
                        .unwrap_or(false);

                    if pointer_released {
                        if is_compatible {
                            if let Some(item) = dragging_item.take() {
                                let drop_x = pointer_pos.map(|p| p.x).unwrap_or(lane.min.x);
                                let drop_frame =
                                    ((drop_x - lane.min.x) / state.zoom).max(0.0) as i64;
                                pending_drop.track_id = Some(track.id);
                                pending_drop.drop_frame = drop_frame;
                                pending_drop.media_path = Some(item.path.clone());
                            }
                        } else {
                            *dragging_item = None;
                        }
                    }
                }

                // ── Draw clips on this track ──────────────────────────────
                let clip_text_color = Color32::WHITE;
                let lane_resp =
                    ui.interact(lane, egui::Id::new(("lane", track.id)), Sense::click());
                let mut clicked_a_clip = false;

                for idx in 0..project.clips.len() {
                    if project.clips.track_id_at(idx) != track.id {
                        continue;
                    }

                    let clip_id = project.clips.clip_id_at(idx);
                    let pts_in = project.clips.pts_in_at(idx);
                    let pts_out = project.clips.pts_out_at(idx);
                    let frame_in = project.pts_to_frame(pts_in);
                    let frame_out = project.pts_to_frame(pts_out);
                    let clip_w_px = ((frame_out - frame_in) as f32 * state.zoom).max(4.0);

                    let is_selected = state.selected_clip == Some(idx);
                    let is_being_dragged = state
                        .drag
                        .as_ref()
                        .map(|d| d.clip_id == clip_id)
                        .unwrap_or(false);
                    let is_being_resized = state
                        .resize
                        .as_ref()
                        .map(|r| r.clip_id == clip_id)
                        .unwrap_or(false);

                    let x_in = GUTTER_W + frame_in as f32 * state.zoom;
                    let clip_rect = Rect::from_min_size(
                        lane.min + egui::vec2(x_in - GUTTER_W, CLIP_H_PAD),
                        Vec2::new(clip_w_px, track_h - CLIP_H_PAD * 2.0),
                    );

                    // ── Interaction sense ─────────────────────────────────
                    let clip_resp = ui.interact(
                        clip_rect,
                        egui::Id::new(("clip", clip_id.index())),
                        Sense::click_and_drag(),
                    );

                    clip_resp.context_menu(|ui| {
                        if ui.button("Delete").clicked() {
                            pending_clip_actions.push(ClipAction::Delete(clip_id));
                            ui.close_menu();
                        }
                        if ui.button("Ripple Delete").clicked() {
                            pending_clip_actions.push(ClipAction::RippleDelete(clip_id));
                            ui.close_menu();
                        }
                        if ui.button("Duplicate").clicked() {
                            pending_clip_actions.push(ClipAction::Duplicate(clip_id));
                            ui.close_menu();
                        }

                        let playhead_pts = project.frame_to_pts(state.playhead_frame);
                        let is_playhead_inside = playhead_pts > pts_in && playhead_pts < pts_out;

                        if ui
                            .add_enabled(is_playhead_inside, egui::Button::new("Split at Playhead"))
                            .clicked()
                        {
                            pending_clip_actions.push(ClipAction::SplitAtPlayhead(clip_id));
                            ui.close_menu();
                        }
                    });

                    // Click → select
                    if clip_resp.clicked() && !is_being_dragged && !is_being_resized {
                        state.selected_clip = Some(idx);
                        clicked_a_clip = true;
                    }

                    // ── Resize / Drag Cursor ──
                    let edge_width = 12.0;
                    if clip_resp.hovered() && !is_being_dragged && !is_being_resized {
                        if let Some(pos) = clip_resp.hover_pos() {
                            if pos.x - clip_rect.left() < edge_width {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                            } else if clip_rect.right() - pos.x < edge_width {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                            }
                        }
                    }

                    // ── Effect drop on clip ───────────────────────────────
                    let is_effect_hover = is_effect_drag
                        && pointer_pos.map_or(false, |p| clip_rect.contains(p));

                    if is_effect_hover && pointer_released {
                        if let Some(item) = dragging_item.take() {
                            let mut eff = project.clips.effects_at(idx);
                            let path_str = item.path.to_string_lossy();
                            if path_str.ends_with("blur") {
                                eff.blur_enabled = true;
                            } else if path_str.ends_with("sharpen") {
                                eff.sharpen_enabled = true;
                            } else if path_str.ends_with("vignette") {
                                eff.vignette_enabled = true;
                            } else if path_str.ends_with("chroma_key") {
                                eff.chroma_key_enabled = true;
                            } else if path_str.ends_with("color") {
                                eff.color_enabled = true;
                            }
                            history.record(project);
                            project.clips.set_effects_at(idx, eff);
                            state.selected_clip = Some(idx);
                            clicked_a_clip = true;
                        }
                    }

                    // Drag start
                    if clip_resp.drag_started()
                        && state.drag.is_none()
                        && state.resize.is_none()
                        && dragging_item.is_none()
                    {
                        let grab_x = ui
                            .ctx()
                            .input(|i| i.pointer.press_origin())
                            .map(|p| p.x)
                            .unwrap_or_else(|| pointer_pos.map(|p| p.x).unwrap_or(clip_rect.min.x));
                        if grab_x - clip_rect.left() < edge_width {
                            state.resize = Some(ClipResize {
                                clip_id,
                                orig_idx: idx,
                                edge: ResizeEdge::Left,
                                orig_pts_in: pts_in,
                                orig_pts_out: pts_out,
                            });
                        } else if clip_rect.right() - grab_x < edge_width {
                            state.resize = Some(ClipResize {
                                clip_id,
                                orig_idx: idx,
                                edge: ResizeEdge::Right,
                                orig_pts_in: pts_in,
                                orig_pts_out: pts_out,
                            });
                        } else {
                            let grab_offset_px = (grab_x - clip_rect.min.x).max(0.0);
                            state.drag = Some(ClipDrag {
                                clip_id,
                                orig_idx: idx,
                                grab_offset_px,
                                orig_track: track.id,
                            });
                        }
                        state.selected_clip = Some(idx);
                        clicked_a_clip = true;
                    }

                    // ── Visual appearance ─────────────────────────────────
                    let alpha: u8 = if is_being_dragged || is_being_resized {
                        60
                    } else {
                        255
                    };

                    let base_color = if clip_resp.hovered()
                        && !is_selected
                        && !is_being_dragged
                        && !is_being_resized
                    {
                        Color32::from_rgba_unmultiplied(
                            (stripe_color.r() as u16 + 25).min(255) as u8,
                            (stripe_color.g() as u16 + 25).min(255) as u8,
                            (stripe_color.b() as u16 + 25).min(255) as u8,
                            alpha,
                        )
                    } else {
                        Color32::from_rgba_unmultiplied(
                            stripe_color.r(),
                            stripe_color.g(),
                            stripe_color.b(),
                            alpha,
                        )
                    };

                    ui.painter().rect_filled(clip_rect, 3.0, base_color);

                    // If an effect is being hovered over this clip, draw glowing effect border
                    if is_effect_hover {
                        ui.painter().rect_stroke(
                            clip_rect,
                            3.0,
                            egui::Stroke::new(2.5, Color32::from_rgb(255, 200, 0)),
                        );
                        ui.painter().rect_filled(
                            clip_rect,
                            3.0,
                            Color32::from_rgba_unmultiplied(255, 200, 0, 45),
                        );
                    }

                    // Top strip
                    let top_strip =
                        Rect::from_min_size(clip_rect.min, Vec2::new(clip_rect.width(), 3.0));
                    ui.painter().rect_filled(
                        top_strip,
                        3.0,
                        Color32::from_rgba_unmultiplied(
                            255,
                            255,
                            255,
                            if is_being_dragged || is_being_resized {
                                15
                            } else {
                                40
                            },
                        ),
                    );

                    // ── Waveform display (audio tracks only) ──────────────────
                    if matches!(track.kind, TrackKind::Audio { .. }) {
                        let source_id = project.clips.source_id_at(idx);
                        // Request waveform extraction on first encounter
                        if let Some(path) = project
                            .sources
                            .read()
                            .unwrap()
                            .path(source_id)
                            .map(|p| p.as_ref().to_path_buf())
                        {
                            waveform_cache.request(source_id, path);
                        }
                        // Draw if available
                        if let Some(wf) = waveform_cache.get(source_id) {
                            if !wf.peaks.is_empty() && clip_rect.width() > 4.0 {
                                let source_in_pts = project.clips.source_in_at(idx);
                                let clip_dur_pts = pts_out - pts_in;
                                // 100 Hz peaks, 90 kHz timebase → 900 pts per peak
                                const PEAK_PTS: f64 = 900.0;
                                let start_peak = (source_in_pts as f64 / PEAK_PTS) as usize;
                                let peaks_needed =
                                    (clip_dur_pts as f64 / PEAK_PTS).ceil() as usize + 1;
                                let peaks_slice = &wf.peaks[start_peak.min(wf.peaks.len())
                                    ..(start_peak + peaks_needed).min(wf.peaks.len())];

                                let wave_rect = clip_rect.shrink2(egui::vec2(0.0, 4.0));
                                let mid_y = wave_rect.center().y;
                                let half_h = wave_rect.height() * 0.45;

                                let clipped_painter = ui.painter().with_clip_rect(clip_rect);
                                if !peaks_slice.is_empty() {
                                    let px_per_peak = clip_rect.width() / peaks_slice.len() as f32;
                                    for (i, &peak) in peaks_slice.iter().enumerate() {
                                        let x = clip_rect.min.x
                                            + i as f32 * px_per_peak
                                            + px_per_peak * 0.5;
                                        let h = (peak * half_h).max(1.0);
                                        let wave_color = Color32::from_rgba_unmultiplied(
                                            200,
                                            230,
                                            255,
                                            if is_being_dragged || is_being_resized {
                                                60
                                            } else {
                                                140
                                            },
                                        );
                                        clipped_painter.line_segment(
                                            [egui::pos2(x, mid_y - h), egui::pos2(x, mid_y + h)],
                                            egui::Stroke::new(1.0, wave_color),
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Selection border and Resize handles
                    if is_selected && !is_being_dragged && !is_being_resized {
                        ui.painter().rect_stroke(
                            clip_rect,
                            3.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(0, 200, 255)),
                        );
                        ui.painter().rect_filled(
                            clip_rect,
                            3.0,
                            Color32::from_rgba_unmultiplied(255, 255, 255, 30),
                        );

                        // Draw visual resize handles on the edges if the clip is wide enough
                        if clip_rect.width() > 16.0 {
                            let handle_w = 4.0;
                            let handle_h = clip_rect.height() * 0.4;
                            let handle_y = clip_rect.center().y - handle_h * 0.5;

                            // Left handle
                            let left_handle = Rect::from_min_size(
                                egui::pos2(clip_rect.left() + 2.0, handle_y),
                                Vec2::new(handle_w, handle_h),
                            );
                            ui.painter().rect_filled(left_handle, 2.0, Color32::WHITE);

                            // Right handle
                            let right_handle = Rect::from_min_size(
                                egui::pos2(clip_rect.right() - handle_w - 2.0, handle_y),
                                Vec2::new(handle_w, handle_h),
                            );
                            ui.painter().rect_filled(right_handle, 2.0, Color32::WHITE);
                        }
                    }

                    // Label
                    if clip_rect.width() > 30.0 && !is_being_dragged && !is_being_resized {
                        let sources = project.sources.read().unwrap();
                        let source_id = project.clips.source_id_at(idx);
                        let clip_name = sources
                            .path(source_id)
                            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                            .unwrap_or_else(|| format!("Clip {}", idx));

                        let text_clip_rect = clip_rect.shrink(2.0); // Slight padding
                        let clipped_painter = ui.painter().with_clip_rect(text_clip_rect);
                        clipped_painter.text(
                            clip_rect.min + egui::vec2(5.0, clip_rect.height() * 0.5),
                            Align2::LEFT_CENTER,
                            clip_name,
                            egui::FontId::proportional(10.0),
                            clip_text_color,
                        );
                    }
                } // end clips loop

                // Deselect on empty lane click
                if lane_resp.clicked() && !clicked_a_clip && state.drag.is_none() {
                    state.selected_clip = None;
                }
                } // end per-track loop

                // Draw add track button
                if current_kind_order == 0 {
                    let btn_h = 24.0;
                    let (row_rect, _) =
                        ui.allocate_exact_size(Vec2::new(canvas_w + GUTTER_W, btn_h), Sense::hover());
                    let gutter = Rect::from_min_size(row_rect.min, Vec2::new(GUTTER_W, btn_h));
                    ui.painter().rect_filled(gutter, 0.0, Color32::from_rgb(28, 28, 28));
                    let btn_rect = gutter.shrink2(egui::vec2(8.0, 4.0));
                    let resp = ui.interact(btn_rect, egui::Id::new("add_video_btn"), Sense::click());
                    let btn_color = if resp.hovered() { Color32::from_rgb(60, 60, 60) } else { Color32::from_rgb(40, 40, 40) };
                    ui.painter().rect_filled(btn_rect, 4.0, btn_color);
                    ui.painter().text(
                        btn_rect.center(), Align2::CENTER_CENTER, "+",
                        egui::FontId::proportional(14.0),
                        if resp.hovered() { Color32::WHITE } else { Color32::LIGHT_GRAY },
                    );
                    if resp.clicked() {
                        pending_track_mutations.push(TrackMutation::AddVideoTrack);
                    }
                } else if current_kind_order == 1 {
                    let btn_h = 24.0;
                    let (row_rect, _) =
                        ui.allocate_exact_size(Vec2::new(canvas_w + GUTTER_W, btn_h), Sense::hover());
                    let gutter = Rect::from_min_size(row_rect.min, Vec2::new(GUTTER_W, btn_h));
                    ui.painter().rect_filled(gutter, 0.0, Color32::from_rgb(28, 28, 28));
                    let btn_rect = gutter.shrink2(egui::vec2(8.0, 4.0));
                    let resp = ui.interact(btn_rect, egui::Id::new("add_audio_btn"), Sense::click());
                    let btn_color = if resp.hovered() { Color32::from_rgb(60, 60, 60) } else { Color32::from_rgb(40, 40, 40) };
                    ui.painter().rect_filled(btn_rect, 4.0, btn_color);
                    ui.painter().text(
                        btn_rect.center(), Align2::CENTER_CENTER, "+",
                        egui::FontId::proportional(14.0),
                        if resp.hovered() { Color32::WHITE } else { Color32::LIGHT_GRAY },
                    );
                    if resp.clicked() {
                        pending_track_mutations.push(TrackMutation::AddAudioTrack);
                    }
                }
            } // end tracks loop

            // ── Apply pending track mutations ─────────────────────────────
            for mutation in pending_track_mutations {
                match mutation {
                    TrackMutation::ToggleMute(id) => {
                        history.record(project);
                        if let Some(track) = project.tracks.get_mut(id) {
                            track.mute = !track.mute;
                        }
                    }
                    TrackMutation::ToggleSolo(id) => {
                        history.record(project);
                        if let Some(track) = project.tracks.get_mut(id) {
                            track.solo = !track.solo;
                        }
                    }
                    TrackMutation::AddVideoTrack => {
                        history.record(project);
                        let count = project.tracks.iter().filter(|t| matches!(t.kind, TrackKind::Video)).count();
                        let _ = project.add_video_track(format!("Video {}", count + 1));
                    }
                    TrackMutation::AddAudioTrack => {
                        history.record(project);
                        let count = project.tracks.iter().filter(|t| matches!(t.kind, TrackKind::Audio { .. })).count();
                        let _ = project.add_audio_track(format!("Audio {}", count + 1));
                    }
                }
            }

            // ── Apply pending clip actions ────────────────────────────────
            for action in pending_clip_actions {
                state.selected_clip = None;
                match action {
                    ClipAction::Delete(id) => {
                        let is_audio = project
                            .clips
                            .index_of(id)
                            .and_then(|idx| {
                                let track_id = project.clips.track_id_at(idx);
                                project.tracks.get(track_id)
                            })
                            .map_or(false, |t| {
                                matches!(t.kind, nexir::timeline::track::TrackKind::Audio { .. })
                            });

                        let linked_id = if is_audio {
                            None
                        } else {
                            find_linked_clip(project, id)
                        };
                        history.record(project);
                        let _ = remove_clip(&mut project.clips, id);
                        if let Some(l_id) = linked_id {
                            let _ = remove_clip(&mut project.clips, l_id);
                        }
                        if state.drag.as_ref().map_or(false, |d| d.clip_id == id) {
                            state.drag = None;
                        }
                    }
                    ClipAction::RippleDelete(id) => {
                        let is_audio = project
                            .clips
                            .index_of(id)
                            .and_then(|idx| {
                                let track_id = project.clips.track_id_at(idx);
                                project.tracks.get(track_id)
                            })
                            .map_or(false, |t| {
                                matches!(t.kind, nexir::timeline::track::TrackKind::Audio { .. })
                            });

                        let linked_id = if is_audio {
                            None
                        } else {
                            find_linked_clip(project, id)
                        };
                        history.record(project);
                        let _ = project.ripple_remove_clip(id);
                        if let Some(l_id) = linked_id {
                            let _ = project.ripple_remove_clip(l_id);
                        }
                        if state.drag.as_ref().map_or(false, |d| d.clip_id == id) {
                            state.drag = None;
                        }
                    }
                    ClipAction::Duplicate(id) => {
                        let linked_id = find_linked_clip(project, id);
                        history.record(project);
                        let _ = nexir::timeline::mutation::duplicate_clip(&mut project.clips, id);
                        if let Some(l_id) = linked_id {
                            let _ =
                                nexir::timeline::mutation::duplicate_clip(&mut project.clips, l_id);
                        }
                    }
                    ClipAction::SplitAtPlayhead(id) => {
                        let playhead_pts = project.frame_to_pts(state.playhead_frame);
                        let linked_id = find_linked_clip(project, id);
                        history.record(project);
                        let _ = nexir::timeline::mutation::split_clip(
                            &mut project.clips,
                            id,
                            playhead_pts,
                        );
                        if let Some(l_id) = linked_id {
                            let _ = nexir::timeline::mutation::split_clip(
                                &mut project.clips,
                                l_id,
                                playhead_pts,
                            );
                        }
                    }
                }
            }

            // ── Ghost clip drawn over every track ─────────────────────────
            if let Some(ref drag) = state.drag {
                if let Some(cursor) = pointer_pos {
                    // Determine which lane (if any) the cursor is over
                    let target_lane = track_lane_rects.iter().find(|(_, r)| r.contains(cursor));

                    // Clip width in pixels
                    let orig_idx = drag.orig_idx;
                    let pts_in = project.clips.pts_in_at(orig_idx);
                    let pts_out = project.clips.pts_out_at(orig_idx);
                    let clip_w_px = ((project.pts_to_frame(pts_out) - project.pts_to_frame(pts_in))
                        as f32
                        * state.zoom)
                        .max(4.0);

                    if let Some(&(_, lane_rect)) = target_lane {
                        let mut new_frame = ((cursor.x - drag.grab_offset_px - lane_rect.min.x)
                            / state.zoom)
                            .max(0.0) as i64;
                        let duration_frames = project.pts_to_frame(pts_out - pts_in);
                        let snap_threshold_px = 8.0;
                        let snap_threshold_frames =
                            (snap_threshold_px / state.zoom).max(1.0) as i64;
                        let mut best_snap_diff = i64::MAX;
                        let mut best_snap_target = None;

                        let playhead = state.playhead_frame;
                        for i in 0..project.clips.len() {
                            let other_id = project.clips.clip_id_at(i);
                            if other_id == drag.clip_id {
                                continue;
                            }
                            let other_in = project.pts_to_frame(project.clips.pts_in_at(i));
                            let other_out = project.pts_to_frame(project.clips.pts_out_at(i));

                            let diff_left_in = (new_frame - other_in).abs();
                            if diff_left_in < best_snap_diff {
                                best_snap_diff = diff_left_in;
                                best_snap_target = Some(other_in);
                            }
                            let diff_left_out = (new_frame - other_out).abs();
                            if diff_left_out < best_snap_diff {
                                best_snap_diff = diff_left_out;
                                best_snap_target = Some(other_out);
                            }
                            let diff_right_in = (new_frame + duration_frames - other_in).abs();
                            if diff_right_in < best_snap_diff {
                                best_snap_diff = diff_right_in;
                                best_snap_target = Some(other_in - duration_frames);
                            }
                            let diff_right_out = (new_frame + duration_frames - other_out).abs();
                            if diff_right_out < best_snap_diff {
                                best_snap_diff = diff_right_out;
                                best_snap_target = Some(other_out - duration_frames);
                            }
                        }

                        let diff_left_ph = (new_frame - playhead).abs();
                        if diff_left_ph < best_snap_diff {
                            best_snap_diff = diff_left_ph;
                            best_snap_target = Some(playhead);
                        }
                        let diff_right_ph = (new_frame + duration_frames - playhead).abs();
                        if diff_right_ph < best_snap_diff {
                            best_snap_diff = diff_right_ph;
                            best_snap_target = Some(playhead - duration_frames);
                        }

                        let mut is_snapped = false;
                        if best_snap_diff <= snap_threshold_frames {
                            if let Some(target) = best_snap_target {
                                new_frame = target.max(0);
                                is_snapped = true;
                            }
                        }

                        let ghost_x = lane_rect.min.x + new_frame as f32 * state.zoom;
                        let track_h = lane_rect.height();
                        let ghost_rect = Rect::from_min_size(
                            egui::pos2(ghost_x, lane_rect.min.y + CLIP_H_PAD),
                            Vec2::new(clip_w_px, track_h - CLIP_H_PAD * 2.0),
                        );

                        // Ghost fill — semi-transparent cyan
                        ui.painter().rect_filled(
                            ghost_rect,
                            3.0,
                            Color32::from_rgba_unmultiplied(0, 180, 255, 80),
                        );
                        ui.painter().rect_stroke(
                            ghost_rect,
                            3.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(0, 220, 255)),
                        );

                        // Snap indicator: frame label at ghost left edge
                        let ghost_secs = new_frame / fps;
                        let ghost_ff = new_frame % fps;
                        ui.painter().text(
                            ghost_rect.left_top() + egui::vec2(4.0, 2.0),
                            Align2::LEFT_TOP,
                            format!("{}:{:02}", ghost_secs, ghost_ff),
                            egui::FontId::proportional(9.0),
                            Color32::WHITE,
                        );

                        // Vertical snap line at ghost left edge
                        let snap_color = if is_snapped {
                            Color32::from_rgba_unmultiplied(255, 120, 0, 225)
                        } else {
                            Color32::from_rgba_unmultiplied(0, 220, 255, 160)
                        };
                        ui.painter().line_segment(
                            [
                                egui::pos2(ghost_x, lane_rect.min.y),
                                egui::pos2(ghost_x, lane_rect.max.y),
                            ],
                            egui::Stroke::new(if is_snapped { 1.5 } else { 1.0 }, snap_color),
                        );
                    }
                }

                // ── Release: commit the move ──────────────────────────────
                if pointer_released {
                    if let Some(cursor) = pointer_pos {
                        let target = track_lane_rects.iter().find(|(_, r)| r.contains(cursor));

                        if let Some(&(target_track_id, lane_rect)) = target {
                            let mut new_frame = ((cursor.x - drag.grab_offset_px - lane_rect.min.x)
                                / state.zoom)
                                .max(0.0) as i64;

                            // Snapping calculation
                            let orig_idx = drag.orig_idx;
                            let pts_in = project.clips.pts_in_at(orig_idx);
                            let pts_out = project.clips.pts_out_at(orig_idx);
                            let duration_frames = project.pts_to_frame(pts_out - pts_in);
                            let snap_threshold_px = 8.0;
                            let snap_threshold_frames =
                                (snap_threshold_px / state.zoom).max(1.0) as i64;
                            let mut best_snap_diff = i64::MAX;
                            let mut best_snap_target = None;

                            let playhead = state.playhead_frame;
                            for i in 0..project.clips.len() {
                                let other_id = project.clips.clip_id_at(i);
                                if other_id == drag.clip_id {
                                    continue;
                                }
                                let other_in = project.pts_to_frame(project.clips.pts_in_at(i));
                                let other_out = project.pts_to_frame(project.clips.pts_out_at(i));

                                let diff_left_in = (new_frame - other_in).abs();
                                if diff_left_in < best_snap_diff {
                                    best_snap_diff = diff_left_in;
                                    best_snap_target = Some(other_in);
                                }
                                let diff_left_out = (new_frame - other_out).abs();
                                if diff_left_out < best_snap_diff {
                                    best_snap_diff = diff_left_out;
                                    best_snap_target = Some(other_out);
                                }
                                let diff_right_in = (new_frame + duration_frames - other_in).abs();
                                if diff_right_in < best_snap_diff {
                                    best_snap_diff = diff_right_in;
                                    best_snap_target = Some(other_in - duration_frames);
                                }
                                let diff_right_out =
                                    (new_frame + duration_frames - other_out).abs();
                                if diff_right_out < best_snap_diff {
                                    best_snap_diff = diff_right_out;
                                    best_snap_target = Some(other_out - duration_frames);
                                }
                            }

                            let diff_left_ph = (new_frame - playhead).abs();
                            if diff_left_ph < best_snap_diff {
                                best_snap_diff = diff_left_ph;
                                best_snap_target = Some(playhead);
                            }
                            let diff_right_ph = (new_frame + duration_frames - playhead).abs();
                            if diff_right_ph < best_snap_diff {
                                best_snap_diff = diff_right_ph;
                                best_snap_target = Some(playhead - duration_frames);
                            }

                            if best_snap_diff <= snap_threshold_frames {
                                if let Some(target) = best_snap_target {
                                    new_frame = target.max(0);
                                }
                            }

                            pending_clip_move = Some(PendingClipMove {
                                clip_id: drag.clip_id,
                                new_track: target_track_id,
                                new_frame,
                            });
                        }
                        // else: dropped outside any lane — cancel (clip stays put)
                    }
                    state.drag = None;
                }

                ui.ctx().request_repaint();
            }

            if let Some(ref resize) = state.resize {
                if let Some(cursor) = pointer_pos {
                    if let Some(&(_, lane_rect)) = track_lane_rects
                        .iter()
                        .find(|(t, _)| *t == project.clips.track_id_at(resize.orig_idx))
                    {
                        let track_h = lane_rect.height();
                        let mut cursor_frame =
                            ((cursor.x - lane_rect.min.x) / state.zoom).max(0.0) as i64;

                        // Resizing snap logic
                        let snap_threshold_px = 8.0;
                        let snap_threshold_frames =
                            (snap_threshold_px / state.zoom).max(1.0) as i64;
                        let mut best_snap_diff = i64::MAX;
                        let mut best_snap_target = None;

                        let playhead = state.playhead_frame;
                        let diff_ph = (cursor_frame - playhead).abs();
                        if diff_ph < best_snap_diff {
                            best_snap_diff = diff_ph;
                            best_snap_target = Some(playhead);
                        }

                        for i in 0..project.clips.len() {
                            let other_id = project.clips.clip_id_at(i);
                            if other_id == resize.clip_id {
                                continue;
                            }
                            let other_in = project.pts_to_frame(project.clips.pts_in_at(i));
                            let other_out = project.pts_to_frame(project.clips.pts_out_at(i));

                            let diff_in = (cursor_frame - other_in).abs();
                            if diff_in < best_snap_diff {
                                best_snap_diff = diff_in;
                                best_snap_target = Some(other_in);
                            }
                            let diff_out = (cursor_frame - other_out).abs();
                            if diff_out < best_snap_diff {
                                best_snap_diff = diff_out;
                                best_snap_target = Some(other_out);
                            }
                        }

                        let mut is_snapped = false;
                        if best_snap_diff <= snap_threshold_frames {
                            if let Some(target) = best_snap_target {
                                cursor_frame = target;
                                is_snapped = true;
                            }
                        }

                        let mut new_pts_in = resize.orig_pts_in;
                        let mut new_pts_out = resize.orig_pts_out;

                        let min_dur = project.frame_to_pts(1);
                        match resize.edge {
                            ResizeEdge::Left => {
                                new_pts_in = project
                                    .frame_to_pts(cursor_frame)
                                    .min(resize.orig_pts_out - min_dur);
                            }
                            ResizeEdge::Right => {
                                new_pts_out = project
                                    .frame_to_pts(cursor_frame)
                                    .max(resize.orig_pts_in + min_dur);
                            }
                        }

                        let frame_in = project.pts_to_frame(new_pts_in);
                        let frame_out = project.pts_to_frame(new_pts_out);
                        let clip_w_px = ((frame_out - frame_in) as f32 * state.zoom).max(4.0);
                        let ghost_x = lane_rect.min.x + frame_in as f32 * state.zoom;

                        let ghost_rect = Rect::from_min_size(
                            egui::pos2(ghost_x, lane_rect.min.y + CLIP_H_PAD),
                            Vec2::new(clip_w_px, track_h - CLIP_H_PAD * 2.0),
                        );

                        ui.painter().rect_filled(
                            ghost_rect,
                            3.0,
                            Color32::from_rgba_unmultiplied(255, 180, 0, 80),
                        );
                        ui.painter().rect_stroke(
                            ghost_rect,
                            3.0,
                            egui::Stroke::new(2.0, Color32::from_rgb(255, 200, 0)),
                        );

                        let ghost_frame = if matches!(resize.edge, ResizeEdge::Left) {
                            frame_in
                        } else {
                            frame_out
                        };
                        let ghost_secs = ghost_frame / fps;
                        let ghost_ff = ghost_frame % fps;
                        let align = if matches!(resize.edge, ResizeEdge::Left) {
                            Align2::LEFT_TOP
                        } else {
                            Align2::RIGHT_TOP
                        };
                        let pos = if matches!(resize.edge, ResizeEdge::Left) {
                            ghost_rect.left_top() + egui::vec2(4.0, 2.0)
                        } else {
                            ghost_rect.right_top() + egui::vec2(-4.0, 2.0)
                        };

                        ui.painter().text(
                            pos,
                            align,
                            format!("{}:{:02}", ghost_secs, ghost_ff),
                            egui::FontId::proportional(9.0),
                            Color32::WHITE,
                        );

                        // Visual snap line for resize
                        if is_snapped {
                            let snap_x = lane_rect.min.x + cursor_frame as f32 * state.zoom;
                            ui.painter().line_segment(
                                [
                                    egui::pos2(snap_x, lane_rect.min.y),
                                    egui::pos2(snap_x, lane_rect.max.y),
                                ],
                                egui::Stroke::new(
                                    1.5,
                                    Color32::from_rgba_unmultiplied(255, 120, 0, 225),
                                ),
                            );
                        }
                    }
                }

                if pointer_released {
                    if let Some(cursor) = pointer_pos {
                        if let Some(&(_, lane_rect)) = track_lane_rects
                            .iter()
                            .find(|(t, _)| *t == project.clips.track_id_at(resize.orig_idx))
                        {
                            let mut cursor_frame =
                                ((cursor.x - lane_rect.min.x) / state.zoom).max(0.0) as i64;

                            // Snapping logic for commit
                            let snap_threshold_px = 8.0;
                            let snap_threshold_frames =
                                (snap_threshold_px / state.zoom).max(1.0) as i64;
                            let mut best_snap_diff = i64::MAX;
                            let mut best_snap_target = None;

                            let playhead = state.playhead_frame;
                            let diff_ph = (cursor_frame - playhead).abs();
                            if diff_ph < best_snap_diff {
                                best_snap_diff = diff_ph;
                                best_snap_target = Some(playhead);
                            }

                            for i in 0..project.clips.len() {
                                let other_id = project.clips.clip_id_at(i);
                                if other_id == resize.clip_id {
                                    continue;
                                }
                                let other_in = project.pts_to_frame(project.clips.pts_in_at(i));
                                let other_out = project.pts_to_frame(project.clips.pts_out_at(i));

                                let diff_in = (cursor_frame - other_in).abs();
                                if diff_in < best_snap_diff {
                                    best_snap_diff = diff_in;
                                    best_snap_target = Some(other_in);
                                }
                                let diff_out = (cursor_frame - other_out).abs();
                                if diff_out < best_snap_diff {
                                    best_snap_diff = diff_out;
                                    best_snap_target = Some(other_out);
                                }
                            }

                            if best_snap_diff <= snap_threshold_frames {
                                if let Some(target) = best_snap_target {
                                    cursor_frame = target;
                                }
                            }

                            let new_pts = project.frame_to_pts(cursor_frame);
                            pending_clip_resize = Some(PendingClipResize {
                                clip_id: resize.clip_id,
                                edge: resize.edge.clone(),
                                new_pts,
                            });
                        }
                    }
                    state.resize = None;
                }
                ui.ctx().request_repaint();
            }

            // ── Apply any pending clip insertion (from media-pool drop) ───
            if let (Some(track_id), Some(path)) = (pending_drop.track_id, pending_drop.media_path) {
                if !project.sources.read().unwrap().path_registered(&path) {
                    let (vid_info, aud_info) = {
                        if let Ok(demuxer) = nexir::io::demuxer::Demuxer::open(&path) {
                            let project_tb = project.settings.timebase;
                            let is_still_image =
                                nexir::timeline::source::is_still_image_path(&path);
                            let vi = demuxer.video_stream.as_ref().map(|s| {
                                nexir::timeline::source::VideoStreamInfo {
                                    width: s.width.unwrap_or(project.settings.width),
                                    height: s.height.unwrap_or(project.settings.height),
                                    frame_rate: if is_still_image {
                                        nexir::timeline::rational::Rational { num: 0, den: 1 }
                                    } else {
                                        s.frame_rate.unwrap_or(
                                            nexir::timeline::rational::Rational { num: 30, den: 1 },
                                        )
                                    },
                                    pixel_fmt: if s.color_info.bit_depth >= 10 {
                                        nexir::timeline::source::PixelFormat::P010
                                    } else {
                                        nexir::timeline::source::PixelFormat::Yuv420p
                                    },
                                    color_info: s.color_info,
                                    duration_pts: if is_still_image {
                                        0
                                    } else {
                                        project_tb.from_pts(s.duration, s.time_base)
                                    },
                                    is_vfr: !is_still_image && s.is_vfr,
                                    time_base: s.time_base,
                                    rotation: nexir::timeline::source::VideoRotation::None,
                                }
                            });
                            let ai = demuxer.audio_stream.as_ref().map(|s| {
                                nexir::timeline::source::AudioStreamInfo {
                                    sample_rate: 48000,
                                    channels: 2,
                                    sample_fmt:
                                        nexir::timeline::source::SampleFormat::F32Interleaved,
                                    duration_pts: project_tb.from_pts(s.duration, s.time_base),
                                }
                            });
                            (vi, ai)
                        } else {
                            (None, None)
                        }
                    };
                    project.register_source(path.clone(), vid_info, aud_info);
                    if !path.to_string_lossy().starts_with("nexir://") {
                        still_cache.lock().unwrap().evict(&path);
                        still_cache.lock().unwrap().probe_decode(&path);
                    }
                }
                let source_id = project.sources.read().unwrap().id_for_path(&path);
                if let Some(source_id) = source_id {
                    let mut duration = project.frame_to_pts(fps * 5);
                    {
                        let sources = project.sources.read().unwrap();
                        if let Ok(v) = sources.video_info(source_id) {
                            if v.duration_pts > 0 {
                                duration = v.duration_pts;
                            }
                        }
                        if let Ok(a) = sources.audio_info(source_id) {
                            if a.duration_pts > 0 {
                                duration = a.duration_pts;
                            }
                        }
                    }
                    let pts_in = project.frame_to_pts(pending_drop.drop_frame);
                    let pts_out = pts_in + duration;

                    history.record(project);
                    // Log clip insertion details so we can trace PNG vs JPG behavior.
                    log::debug!(
                        "Timeline: inserting clip from path={:?} source_id={:?} track_id={:?} pts_in={} pts_out={}",
                        path,
                        source_id,
                        track_id,
                        pts_in,
                        pts_out
                    );
                    let _ = project.insert_clip_ripple(ClipInsertParams {
                        track_id,
                        source_id,
                        kind: if path.to_string_lossy() == "nexir://internal/text" {
                            nexir::timeline::store::ClipKind::Text {
                                text: "Basic Text".to_string(),
                                font_size: 48.0,
                                color: [1.0, 1.0, 1.0, 1.0],
                                stroke_color: None,
                                stroke_width: 2.0,
                                background_color: None,
                                bg_padding: 12.0,
                            }

                        } else {
                            nexir::timeline::store::ClipKind::Video
                        },
                        pts_in,
                        pts_out,
                        source_in: 0,
                        layer_order: 0,
                        opacity: 1.0,
                        transform: ClipTransform::identity(),
                        volume: 1.0,
                        pan: 0.0,
                        audio_muted: false,
                        fade_in_pts: 0,
                        fade_out_pts: 0,
                        speed: 1.0,
                        pitch: 0.0,
                        ..Default::default()
                    });

                    // NOTE: Auto-inserting an audio clip on import is intentionally disabled.
                    // A future voice-separation feature will handle audio placement explicitly.
                }
            }

            // ── Apply pending clip resize ─────────────────────────────────
            if let Some(rsz) = pending_clip_resize {
                if let Some(idx) = project.clips.index_of(rsz.clip_id) {
                    let min_dur = project.frame_to_pts(1);
                    let linked_id = find_linked_clip(project, rsz.clip_id);
                    history.record(project);
                    match rsz.edge {
                        ResizeEdge::Left => {
                            let pts_out = project.clips.pts_out_at(idx);
                            let new_pts_in = rsz.new_pts.min(pts_out - min_dur);
                            let _ = nexir::timeline::mutation::trim_clip_in(
                                &mut project.clips,
                                rsz.clip_id,
                                new_pts_in,
                            );
                            if let Some(l_id) = linked_id {
                                let _ = nexir::timeline::mutation::trim_clip_in(
                                    &mut project.clips,
                                    l_id,
                                    new_pts_in,
                                );
                            }
                        }
                        ResizeEdge::Right => {
                            let pts_in = project.clips.pts_in_at(idx);
                            let new_pts_out = rsz.new_pts.max(pts_in + min_dur);
                            let _ = nexir::timeline::mutation::trim_clip_out(
                                &mut project.clips,
                                rsz.clip_id,
                                new_pts_out,
                            );
                            if let Some(l_id) = linked_id {
                                let _ = nexir::timeline::mutation::trim_clip_out(
                                    &mut project.clips,
                                    l_id,
                                    new_pts_out,
                                );
                            }
                        }
                    }
                }
            }

            // ── Apply pending clip move ───────────────────────────────────
            if let Some(mv) = pending_clip_move {
                let new_pts_in = project.frame_to_pts(mv.new_frame.max(0));

                let orig_idx = project.clips.index_of(mv.clip_id);
                let old_pts_in = orig_idx.map(|i| project.clips.pts_in_at(i));
                let linked_id = find_linked_clip(project, mv.clip_id);

                // Get the clip's original track to decide if we need move_clip or move_clip_to_track
                let orig_track = orig_idx.map(|i| project.clips.track_id_at(i));

                let result = if orig_track == Some(mv.new_track) {
                    history.record(project);
                    project.move_clip(mv.clip_id, new_pts_in)
                } else {
                    history.record(project);
                    project.move_clip_to_track(mv.clip_id, mv.new_track, new_pts_in)
                };

                // Move the linked clip in time by the same delta
                if let (Ok(_), Some(l_id), Some(old_in)) = (&result, linked_id, old_pts_in) {
                    let delta_pts = new_pts_in - old_in;
                    if let Some(l_idx) = project.clips.index_of(l_id) {
                        let l_new_pts_in = project.clips.pts_in_at(l_idx) + delta_pts;
                        let _ = project.move_clip(l_id, l_new_pts_in);
                    }
                }

                // After move, find the new index of the moved clip to keep selection correct
                if let Ok(new_clip_id) = result {
                    let new_idx = project.clips.index_of(new_clip_id);
                    state.selected_clip = new_idx;
                }
            }

            // ── Playhead line ─────────────────────────────────────────────
            let ph_x = GUTTER_W + state.playhead_frame as f32 * state.zoom;
            let ph_top = ruler_rect.min + egui::vec2(ph_x, 0.0);
            let ph_bot = ph_top
                + egui::vec2(
                    0.0,
                    ruler_rect.height() + project.tracks.len() as f32 * 80.0,
                );
            ui.painter().line_segment(
                [ph_top, ph_bot],
                egui::Stroke::new(2.0, Color32::from_rgb(255, 50, 50)),
            );
            ui.painter().circle_filled(
                ph_top + egui::vec2(0.0, RULER_H),
                5.0,
                Color32::from_rgb(255, 50, 50),
            );
        });
}

pub fn find_linked_clip(project: &Project, clip_id: ClipId) -> Option<ClipId> {
    let idx = project.clips.index_of(clip_id)?;
    let source_id = project.clips.source_id_at(idx);
    let pts_in = project.clips.pts_in_at(idx);
    let track_id = project.clips.track_id_at(idx);
    let is_video = matches!(
        project.tracks.get(track_id)?.kind,
        nexir::timeline::track::TrackKind::Video
    );

    for i in 0..project.clips.len() {
        let other_id = project.clips.clip_id_at(i);
        if other_id == clip_id {
            continue;
        }
        if project.clips.source_id_at(i) == source_id && project.clips.pts_in_at(i) == pts_in {
            let other_track_id = project.clips.track_id_at(i);
            if let Some(other_track) = project.tracks.get(other_track_id) {
                let other_is_video =
                    matches!(other_track.kind, nexir::timeline::track::TrackKind::Video);
                if is_video != other_is_video {
                    return Some(other_id);
                }
            }
        }
    }
    None
}
