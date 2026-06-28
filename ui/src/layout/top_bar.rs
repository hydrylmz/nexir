use egui::{Ui, Color32};

/// Persistent state for the top bar (kept alive across frames via egui memory).
#[derive(Default, Clone)]
struct TopBarState {
    about_open: bool,
}

pub fn draw(ui: &mut Ui) {
    // Load/store persistent state in egui memory so it survives across frames.
    let mut state = ui.ctx().data(|d| d.get_temp::<TopBarState>(egui::Id::new("top_bar_state")).unwrap_or_default());

    egui::menu::bar(ui, |ui| {
        ui.menu_button("File", |ui| {
            if ui.button("New Project").clicked() {}
            if ui.button("Open Project...").clicked() {}
            ui.separator();
            if ui.button("Save").clicked() {}
            if ui.button("Save As...").clicked() {}
            ui.separator();
            if ui.button("Exit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
        ui.menu_button("Edit", |ui| {
            if ui.button("Undo").clicked() {}
            if ui.button("Redo").clicked() {}
        });
        ui.menu_button("View", |ui| {
            if ui.button("Reset Layout").clicked() {
                ui.ctx().memory_mut(|mem| *mem = Default::default());
                ui.close_menu();
            }
        });
        ui.menu_button("Help", |ui| {
            if ui.button("About nexir").clicked() {
                state.about_open = true;
                ui.close_menu();
            }
        });

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // CapCut style bright blue export button
            let export_btn = egui::Button::new(
                egui::RichText::new("Export").color(Color32::WHITE)
            ).fill(Color32::from_rgb(0, 153, 255));

            if ui.add(export_btn).clicked() {
                // Open export modal
            }

            ui.add_space(8.0);
            ui.label("nexir Editor");
        });
    });

    // ── About dialog ────────────────────────────────────────────────────────
    if state.about_open {
        egui::Window::new("About nexir")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                ui.add_space(8.0);
                ui.vertical_centered(|ui| {
                    ui.label(egui::RichText::new("nexir").size(28.0).strong().color(Color32::from_rgb(0, 153, 255)));
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Video Editor").size(13.0).color(Color32::GRAY));
                    ui.add_space(12.0);
                    ui.label("A fast, GPU-accelerated non-linear video editor");
                    ui.label("built in Rust with wgpu + egui.");
                    ui.add_space(12.0);
                    ui.label(egui::RichText::new("v0.1.0").monospace().color(Color32::LIGHT_GRAY));
                    ui.add_space(16.0);
                    if ui.button("  Close  ").clicked() {
                        state.about_open = false;
                    }
                    ui.add_space(8.0);
                });
            });
    }

    // Persist state back into egui memory.
    ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new("top_bar_state"), state));
}
