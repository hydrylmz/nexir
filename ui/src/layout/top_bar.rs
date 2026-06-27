use egui::{Ui, Color32};

pub fn draw(ui: &mut Ui) {
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
            if ui.button("Reset Layout").clicked() {}
        });
        ui.menu_button("Help", |ui| {
            if ui.button("About nexir").clicked() {}
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
}
