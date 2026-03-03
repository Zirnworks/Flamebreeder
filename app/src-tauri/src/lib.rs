mod commands;
mod sidecar;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            sidecar::start_sidecar(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::generate_random,
            commands::breed,
            commands::interpolate,
            commands::mutate_genome,
            commands::remap_genome,
            commands::get_genome,
            commands::get_genome_image,
            commands::list_genomes,
            commands::update_genome,
            commands::check_server_health,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
