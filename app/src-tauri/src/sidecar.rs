//! Manages the Python inference server sidecar process.

use std::process::Command;
use std::env;

/// Spawn the Python inference server in the background.
pub fn start_sidecar(_app: tauri::AppHandle) {
    let project_dir = env::var("PRAECEPTOR_PROJECT_DIR")
        .unwrap_or_else(|_| {
            let home = env::var("HOME").unwrap_or_else(|_| "/Users/zirn".to_string());
            format!("{}/Data/Praeceptor/project", home)
        });

    let python = env::var("PRAECEPTOR_PYTHON")
        .unwrap_or_else(|_| format!("{}/breeding/.venv/bin/python3.11", project_dir));

    let checkpoint = env::var("PRAECEPTOR_CHECKPOINT")
        .unwrap_or_else(|_| {
            let home = env::var("HOME").unwrap_or_else(|_| "/Users/zirn".to_string());
            format!("{}/Data/Praeceptor/models/stylegan2-ada-fractals-k30-kimg880.pkl", home)
        });

    let port = env::var("PRAECEPTOR_PORT")
        .unwrap_or_else(|_| "8420".to_string());

    let device = env::var("PRAECEPTOR_DEVICE")
        .unwrap_or_else(|_| "mps".to_string());

    println!("[sidecar] Starting inference server...");
    println!("[sidecar] Python: {}", python);
    println!("[sidecar] Checkpoint: {}", checkpoint);

    match Command::new(&python)
        .arg("-m")
        .arg("breeding.server")
        .arg(&checkpoint)
        .arg("--port")
        .arg(&port)
        .arg("--device")
        .arg(&device)
        .current_dir(&project_dir)
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        .spawn()
    {
        Ok(child) => {
            println!("[sidecar] Server spawned (PID: {})", child.id());
        }
        Err(e) => {
            eprintln!("[sidecar] Failed to start server: {}", e);
            eprintln!("[sidecar] Python: {}", python);
        }
    }
}
