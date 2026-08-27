use crate::history::HistoryState;
use crate::layout::timeline::TimelineState;
use egui::{Align2, Color32, Rect, RichText, Ui, Vec2, pos2};
use nexir::project::Project;
use nexir::timeline::query::ActiveClip;
use nexir::timeline::transform::ClipTransform;

fn clip_corners_ui(
    transform: &ClipTransform,
    clip_w: f32,
    clip_h: f32,
    draw_rect: Rect,
    canvas_w: f32,
    canvas_h: f32,
) -> [egui::Pos2; 4] {
    let m = transform.to_matrix(clip_w, clip_h, canvas_w, canvas_h);

    let map_corner = |u: f32, v: f32| -> egui::Pos2 {
        let ndc_x = u * m[0] + v * m[3] + m[6];
        let ndc_y = u * m[1] + v * m[4] + m[7];

        let nx = (ndc_x + 1.0) * 0.5;
        let ny = (1.0 - ndc_y) * 0.5;

        pos2(
            draw_rect.min.x + nx * draw_rect.width(),
            draw_rect.min.y + ny * draw_rect.height(),
        )
    };

    [
        map_corner(0.0, 0.0), // top-left
        map_corner(1.0, 0.0), // top-right
        map_corner(1.0, 1.0), // bottom-right
        map_corner(0.0, 1.0), // bottom-left
    ]
}

fn is_point_in_quad(p: egui::Pos2, quad: &[egui::Pos2; 4]) -> bool {
    let mut inside = false;
    let mut j = 3;
    for i in 0..4 {
        if ((quad[i].y > p.y) != (quad[j].y > p.y))
            && (p.x
                < (quad[j].x - quad[i].x) * (p.y - quad[i].y) / (quad[j].y - quad[i].y) + quad[i].x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Returned by `viewport::draw`.
pub struct ViewportDrawResult {
    pub size: Vec2,
    /// If the eyedropper was active and the user clicked inside the video area,
    /// this is the normalized UV coordinate (0..1 range) of the click.
    pub eyedropper_pick: Option<egui::Vec2>,
}

/// Draw the preview viewport, preserving the video's aspect ratio via letter-boxing / pillar-boxing.
pub fn draw(
    ui: &mut Ui,
    preview_id: Option<egui::TextureId>,
    _video_width: u32,
    _video_height: u32,
    state: &mut TimelineState,
    project: &mut Project,
    active_clips: &[ActiveClip],
    history: &mut HistoryState,
    eyedropper_active: bool,
) -> ViewportDrawResult {
    let fps = project.settings.frame_rate.num.max(1);
    let f = state.playhead_frame;
    let ff = f % fps;
    let ss = (f / fps) % 60;
    let mm = (f / (fps * 60)) % 60;
    let hh = f / (fps * 3600);
    let timecode = format!("{:02}:{:02}:{:02}:{:02}", hh, mm, ss, ff);

    ui.horizontal(|ui| {
        ui.label(RichText::new("Player").color(Color32::WHITE));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(timecode)
                    .monospace()
                    .color(Color32::LIGHT_GRAY),
            );
        });
    });

    ui.add_space(4.0);

    let available_size = ui.available_size();
    let controls_height = 40.0;
    let viewport_size = Vec2::new(available_size.x, available_size.y - controls_height);

    let (rect, response) = ui.allocate_exact_size(viewport_size, egui::Sense::click());
    ui.painter().rect_filled(rect, 0.0, Color32::BLACK);

    let canvas_w = project.settings.width as f32;
    let canvas_h = project.settings.height as f32;
    let canvas_ar = canvas_w / canvas_h;
    let panel_ar = rect.width() / rect.height();

    let (draw_w, draw_h) = if canvas_ar > panel_ar {
        let w = rect.width();
        (w, w / canvas_ar)
    } else {
        let h = rect.height();
        (h * canvas_ar, h)
    };

    let draw_rect = Rect::from_min_size(
        pos2(
            rect.center().x - draw_w * 0.5,
            rect.center().y - draw_h * 0.5,
        ),
        Vec2::new(draw_w, draw_h),
    );

    if let Some(tex_id) = preview_id {
        let uv = egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        ui.painter().image(tex_id, draw_rect, uv, Color32::WHITE);
    } else {
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            "No Media Selected",
            egui::FontId::proportional(24.0),
            Color32::DARK_GRAY,
        );
    }

    // Eyedropper mode cursor feedback
    if eyedropper_active {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }

    // Eyedropper pick — intercept click before normal hit-testing
    let mut eyedropper_pick: Option<egui::Vec2> = None;
    if eyedropper_active && response.clicked() {
        if let Some(pos) = response.interact_pointer_pos()
            .or_else(|| ui.input(|i| i.pointer.latest_pos()))
            .or_else(|| ui.ctx().pointer_interact_pos())
        {
            if draw_rect.contains(pos) {
                let u = (pos.x - draw_rect.min.x) / draw_rect.width();
                let v = (pos.y - draw_rect.min.y) / draw_rect.height();
                eyedropper_pick = Some(egui::vec2(u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)));
            }
        }
    }

    // Hit-testing (only when eyedropper is not active)
    if !eyedropper_active && response.clicked() && !state.is_interacting_with_clip() {
        if let Some(pos) = ui.ctx().pointer_interact_pos() {
            let mut clicked_idx = None;
            for clip in active_clips.iter().rev() {
                let track_id = project.clips.track_id_at(clip.store_index);
                let is_video_track = project.tracks.get(track_id).map_or(false, |t| {
                    matches!(t.kind, nexir::timeline::track::TrackKind::Video)
                });
                if !is_video_track {
                    continue;
                }

                let (clip_w, clip_h) = {
                    let kind = project.clips.kind_at(clip.store_index);
                    if let nexir::timeline::store::ClipKind::Text {
                        text, font_size, stroke_color, stroke_width, background_color, bg_padding, ..
                    } = kind {
                        let (w, h) = nexir::render::text_renderer::measure_text(
                            text,
                            *font_size,
                            *stroke_width,
                            stroke_color.is_some(),
                            background_color.is_some(),
                            *bg_padding,
                        );
                        (w as f32, h as f32)
                    } else {
                        let source_id = project.clips.source_id_at(clip.store_index);
                        if let Ok(info) = project.sources.read().unwrap().video_info(source_id) {
                            (info.width as f32, info.height as f32)
                        } else {
                            continue;
                        }
                    }
                };

                let transform = project.clips.transform_at(clip.store_index);
                let corners = clip_corners_ui(&transform, clip_w, clip_h, draw_rect, canvas_w, canvas_h);
                if is_point_in_quad(pos, &corners) {
                    clicked_idx = Some(clip.store_index);
                    break;
                }
            }
            state.selected_clip = clicked_idx;
        }
    }

    // Selected clip overlay
    if let Some(idx) = state.selected_clip {
        if active_clips.iter().any(|c| c.store_index == idx) {
            let track_id = project.clips.track_id_at(idx);
            let is_video_track = project.tracks.get(track_id).map_or(false, |t| {
                matches!(t.kind, nexir::timeline::track::TrackKind::Video)
            });
            if is_video_track {
                let (clip_w, clip_h) = {
                    let kind = project.clips.kind_at(idx);
                    if let nexir::timeline::store::ClipKind::Text {
                        text, font_size, stroke_color, stroke_width, background_color, bg_padding, ..
                    } = kind {
                        let (w, h) = nexir::render::text_renderer::measure_text(
                            text,
                            *font_size,
                            *stroke_width,
                            stroke_color.is_some(),
                            background_color.is_some(),
                            *bg_padding,
                        );
                        (w as f32, h as f32)
                    } else {
                        let source_id = project.clips.source_id_at(idx);
                        if let Ok(info) = project.sources.read().unwrap().video_info(source_id) {
                            (info.width as f32, info.height as f32)
                        } else {
                            (1920.0, 1080.0) // Fallback should rarely happen
                        }
                    }
                };

                let transform = *project.clips.transform_at(idx);
                let clip_id = project.clips.clip_id_at(idx);

                let corners = clip_corners_ui(&transform, clip_w, clip_h, draw_rect, canvas_w, canvas_h);

                // Draw outline
                for i in 0..4 {
                    ui.painter().line_segment(
                        [corners[i], corners[(i + 1) % 4]],
                        egui::Stroke::new(2.0, Color32::LIGHT_BLUE),
                    );
                }

                let handle_radius = 6.0;
                let delete_radius = 8.0;
                let handle_rects = corners.map(|corner| {
                    Rect::from_center_size(corner, Vec2::splat(handle_radius * 3.0))
                });
                
                // Rotate handle (above top-center)
                let top_center = corners[0].lerp(corners[1], 0.5);
                let up_vector = (corners[0] - corners[3]).normalized(); 
                let rotate_center = top_center + up_vector * 25.0; // Extend outward above the top edge
                let rotate_rect = Rect::from_center_size(rotate_center, Vec2::splat(handle_radius * 3.0));

                let tr = corners[1];
                let del_center = tr + egui::vec2(12.0, -12.0);
                let del_rect =
                    Rect::from_center_size(del_center, Vec2::splat(delete_radius * 2.5));

                let mut min = corners[0];
                let mut max = corners[0];
                for &corner in &corners[1..] {
                    min.x = min.x.min(corner.x);
                    min.y = min.y.min(corner.y);
                    max.x = max.x.max(corner.x);
                    max.y = max.y.max(corner.y);
                }
                let body_rect = Rect::from_min_max(min, max);
                let move_resp = ui.interact(
                    body_rect,
                    egui::Id::new(("viewport_move", idx)),
                    egui::Sense::drag(),
                );

                if move_resp.drag_started() && state.viewport_resize().is_none() && state.viewport_rotate().is_none() {
                    if let Some(pointer) = ui.ctx().pointer_interact_pos() {
                        let started_on_handle =
                            handle_rects.iter().any(|rect| rect.contains(pointer));
                        if !started_on_handle
                            && !rotate_rect.contains(pointer)
                            && !del_rect.contains(pointer)
                            && is_point_in_quad(pointer, &corners)
                        {
                            history.record(project);
                            state.start_viewport_move(clip_id, pointer, transform);
                        }
                    }
                }

                // Resize Handles
                for (i, (&corner, handle_rect)) in
                    corners.iter().zip(handle_rects.iter()).enumerate()
                {
                    let resize_id = egui::Id::new(("viewport_resize", idx, i));
                    let resp = ui.interact(*handle_rect, resize_id, egui::Sense::drag());

                    if resp.drag_started() {
                        if let Some(pointer) = ui.ctx().pointer_interact_pos() {
                            history.record(project);
                            state.start_viewport_resize(clip_id, i, pointer, transform);
                        }
                    }

                    let color = if resp.hovered() {
                        Color32::WHITE
                    } else {
                        Color32::from_rgb(200, 200, 255)
                    };
                    ui.painter().circle_filled(corner, handle_radius, color);
                    ui.painter().circle_stroke(
                        corner,
                        handle_radius,
                        egui::Stroke::new(1.0, Color32::BLACK),
                    );
                }

                // Draw Rotate Handle
                let rotate_resp = ui.interact(rotate_rect, egui::Id::new(("viewport_rotate", idx)), egui::Sense::drag());
                if rotate_resp.drag_started() {
                    if let Some(pointer) = ui.ctx().pointer_interact_pos() {
                        history.record(project);
                        state.start_viewport_rotate(clip_id, pointer, transform);
                    }
                }
                
                // Draw connecting line to rotate handle
                ui.painter().line_segment(
                    [top_center, rotate_center],
                    egui::Stroke::new(1.5, Color32::LIGHT_BLUE),
                );
                
                let rot_color = if rotate_resp.hovered() { Color32::WHITE } else { Color32::from_rgb(150, 255, 150) };
                ui.painter().circle_filled(rotate_center, handle_radius, rot_color);
                ui.painter().circle_stroke(
                    rotate_center,
                    handle_radius,
                    egui::Stroke::new(1.0, Color32::BLACK),
                );

                if let Some(movement) = state.viewport_move() {
                    if movement.clip_id == clip_id && state.viewport_resize().is_none() {
                        if let Some(pointer) = ui
                            .ctx()
                            .pointer_interact_pos()
                            .or_else(|| ui.ctx().pointer_hover_pos())
                        {
                            let delta = pointer - movement.start_pointer;
                            let mut new_transform = movement.start_transform;
                            new_transform.position[0] += delta.x * canvas_w / draw_rect.width();
                            new_transform.position[1] +=
                                delta.y * canvas_h / draw_rect.height();

                            if state.viewport_snap {
                                // Snap if within 15 pixels screen distance
                                let snap_w = 15.0 * canvas_w / draw_rect.width();
                                let snap_h = 15.0 * canvas_h / draw_rect.height();
                                let mut snapped_x = false;
                                let mut snapped_y = false;

                                if new_transform.position[0].abs() < snap_w {
                                    new_transform.position[0] = 0.0;
                                    snapped_x = true;
                                }
                                if new_transform.position[1].abs() < snap_h {
                                    new_transform.position[1] = 0.0;
                                    snapped_y = true;
                                }

                                if snapped_x {
                                    let x_pos = draw_rect.center().x;
                                    ui.painter().line_segment(
                                        [egui::pos2(x_pos, draw_rect.top()), egui::pos2(x_pos, draw_rect.bottom())],
                                        egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 255, 128)),
                                    );
                                }
                                if snapped_y {
                                    let y_pos = draw_rect.center().y;
                                    ui.painter().line_segment(
                                        [egui::pos2(draw_rect.left(), y_pos), egui::pos2(draw_rect.right(), y_pos)],
                                        egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 255, 128)),
                                    );
                                }
                            }

                            project.clips.set_transform_at(idx, new_transform);
                            ui.ctx().request_repaint();
                        }

                        if ui.input(|input| input.pointer.any_released()) {
                            state.finish_viewport_move();
                        }
                    }
                }

                if let Some(resize) = state.viewport_resize() {
                    if resize.clip_id == clip_id {
                        if let Some(pointer) = ui
                            .ctx()
                            .pointer_interact_pos()
                            .or_else(|| ui.ctx().pointer_hover_pos())
                        {
                            let delta = pointer - resize.start_pointer;
                            let scale_dir = match resize.handle_index {
                                0 => -delta.x - delta.y,
                                1 => delta.x - delta.y,
                                2 => delta.x + delta.y,
                                3 => -delta.x + delta.y,
                                _ => 0.0,
                            };
                            let scale_delta = scale_dir * 0.003;
                            let mut new_transform = resize.start_transform;
                            let sign_x = if resize.start_transform.scale[0] < 0.0 {
                                -1.0
                            } else {
                                1.0
                            };
                            let sign_y = if resize.start_transform.scale[1] < 0.0 {
                                -1.0
                            } else {
                                1.0
                            };
                            new_transform.scale[0] += scale_delta * sign_x;
                            new_transform.scale[1] += scale_delta * sign_y;

                            if new_transform.scale[0].abs() < 0.01 {
                                new_transform.scale[0] = 0.01 * sign_x;
                            }
                            if new_transform.scale[1].abs() < 0.01 {
                                new_transform.scale[1] = 0.01 * sign_y;
                            }

                            project.clips.set_transform_at(idx, new_transform);
                            ui.ctx().request_repaint();
                            if ui.input(|input| input.pointer.any_released()) {
                                state.finish_viewport_resize();
                            }
                        }
                    }
                }

                if let Some(rotate) = state.viewport_rotate() {
                    if rotate.clip_id == clip_id {
                        if let Some(pointer) = ui
                            .ctx()
                            .pointer_interact_pos()
                            .or_else(|| ui.ctx().pointer_hover_pos())
                        {
                            let center_px = (corners[0] + corners[2].to_vec2()) * 0.5;
                            
                            // Angle from center to drag start
                            let start_vec = rotate.start_pointer - center_px;
                            let start_angle = start_vec.y.atan2(start_vec.x);
                            
                            // Angle from center to current pointer
                            let curr_vec = pointer - center_px;
                            let curr_angle = curr_vec.y.atan2(curr_vec.x);
                            
                            let delta_angle = curr_angle - start_angle;
                            
                            let mut new_transform = rotate.start_transform;
                            // Add delta; rotation is counterclockwise in transform so we negate delta to match screen Y-down
                            new_transform.rotation -= delta_angle;
                            
                            project.clips.set_transform_at(idx, new_transform);
                            ui.ctx().request_repaint();
                        }

                        if ui.input(|input| input.pointer.any_released()) {
                            state.finish_viewport_rotate();
                        }
                    }
                }

                // Delete Button
                let del_resp = ui.interact(
                    del_rect,
                    egui::Id::new(("delete", idx)),
                    egui::Sense::click(),
                );

                let del_color = if del_resp.hovered() {
                    Color32::from_rgb(255, 50, 50)
                } else {
                    Color32::RED
                };
                ui.painter()
                    .circle_filled(del_center, delete_radius, del_color);
                ui.painter().circle_stroke(
                    del_center,
                    delete_radius,
                    egui::Stroke::new(1.0, Color32::WHITE),
                );
                ui.painter().line_segment(
                    [
                        del_center + egui::vec2(-3.0, -3.0),
                        del_center + egui::vec2(3.0, 3.0),
                    ],
                    egui::Stroke::new(2.0, Color32::WHITE),
                );
                ui.painter().line_segment(
                    [
                        del_center + egui::vec2(3.0, -3.0),
                        del_center + egui::vec2(-3.0, 3.0),
                    ],
                    egui::Stroke::new(2.0, Color32::WHITE),
                );

                if del_resp.clicked() {
                    history.record(project);
                    let _ = nexir::timeline::mutation::remove_clip(&mut project.clips, clip_id);
                    state.selected_clip = None;
                }
            }
        }
    }

    if state.viewport_resize().is_some() && ui.input(|input| input.pointer.any_released()) {
        state.finish_viewport_resize();
    }
    if state.viewport_move().is_some() && ui.input(|input| input.pointer.any_released()) {
        state.finish_viewport_move();
    }
    if state.viewport_rotate().is_some() && ui.input(|input| input.pointer.any_released()) {
        state.finish_viewport_rotate();
    }

    ui.add_space(8.0);

    ui.horizontal(|ui| {
        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::Center).with_main_justify(true),
            |ui| {
                ui.horizontal(|ui| {
                    if ui.button("⏮").on_hover_text("Go to start").clicked() {
                        state.playhead_frame = 0;
                        state.playing = false;
                    }
                    if ui.button("⏪").on_hover_text("Step back").clicked() {
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
                    if ui.button("⏩").on_hover_text("Step forward").clicked() {
                        state.playing = false;
                        state.playhead_frame += 1;
                    }
                    if ui.button("⏭").on_hover_text("Go to end").clicked() {
                        state.playing = false;
                        state.playhead_frame = project.duration_frames();
                    }
                });
            },
        );
    });

    ViewportDrawResult { size: viewport_size, eyedropper_pick }
}
