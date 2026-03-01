//! Tauri commands that proxy to the Python FastAPI sidecar.

use serde::{Deserialize, Serialize};
use serde_json::Value;

const SIDECAR_URL: &str = "http://127.0.0.1:8420";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GenomeResponse {
    pub id: String,
    pub image_base64: String,
    pub genome: Value,
}

#[tauri::command]
pub async fn generate_random(
    count: u32,
    truncation_psi: Option<f64>,
    class_label: Option<Vec<f64>>,
) -> Result<Vec<GenomeResponse>, String> {
    let psi = truncation_psi.unwrap_or(0.7);
    let client = reqwest::Client::new();
    let mut body = serde_json::json!({ "count": count, "truncation_psi": psi });
    if let Some(cl) = class_label {
        body["class_label"] = serde_json::to_value(cl).unwrap();
    }

    let resp = client
        .post(format!("{}/generate", SIDECAR_URL))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Failed to connect to inference server: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Server error {}: {}", status, text));
    }

    resp.json::<Vec<GenomeResponse>>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}

#[tauri::command]
pub async fn breed(
    parent_a_id: String,
    parent_b_id: String,
    method: String,
    params: Value,
    count: u32,
) -> Result<Vec<GenomeResponse>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/breed", SIDECAR_URL))
        .json(&serde_json::json!({
            "parent_a_id": parent_a_id,
            "parent_b_id": parent_b_id,
            "method": method,
            "params": params,
            "count": count,
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Server error {}: {}", status, text));
    }

    resp.json::<Vec<GenomeResponse>>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}

#[tauri::command]
pub async fn interpolate(
    genome_a_id: String,
    genome_b_id: String,
    steps: u32,
    method: String,
) -> Result<Vec<GenomeResponse>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/interpolate", SIDECAR_URL))
        .json(&serde_json::json!({
            "genome_a_id": genome_a_id,
            "genome_b_id": genome_b_id,
            "steps": steps,
            "method": method,
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Server error {}: {}", status, text));
    }

    resp.json::<Vec<GenomeResponse>>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}

#[tauri::command]
pub async fn mutate_genome(
    genome_id: String,
    rate: f64,
    strength: f64,
) -> Result<GenomeResponse, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/mutate", SIDECAR_URL))
        .json(&serde_json::json!({
            "genome_id": genome_id,
            "rate": rate,
            "strength": strength,
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Server error {}: {}", status, text));
    }

    resp.json::<GenomeResponse>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}

#[tauri::command]
pub async fn remap_genome(
    genome_id: String,
    class_label: Vec<f64>,
    truncation_psi: Option<f64>,
) -> Result<GenomeResponse, String> {
    let psi = truncation_psi.unwrap_or(0.7);
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/remap", SIDECAR_URL))
        .json(&serde_json::json!({
            "genome_id": genome_id,
            "class_label": class_label,
            "truncation_psi": psi,
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Server error {}: {}", status, text));
    }

    resp.json::<GenomeResponse>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}

#[tauri::command]
pub async fn get_genome(genome_id: String) -> Result<GenomeResponse, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/genome/{}", SIDECAR_URL, genome_id))
        .send()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Server error {}: {}", status, text));
    }

    resp.json::<GenomeResponse>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}

#[tauri::command]
pub async fn list_genomes() -> Result<Vec<Value>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/genomes", SIDECAR_URL))
        .send()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Server error {}: {}", status, text));
    }

    resp.json::<Vec<Value>>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}

#[tauri::command]
pub async fn check_server_health() -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .map_err(|e| format!("Failed to create client: {}", e))?;

    let resp = client
        .get(format!("{}/health", SIDECAR_URL))
        .send()
        .await
        .map_err(|e| format!("Server not ready: {}", e))?;

    resp.json::<Value>()
        .await
        .map_err(|e| format!("Failed to parse health response: {}", e))
}

#[tauri::command]
pub async fn update_genome(
    genome_id: String,
    tags: Option<Vec<String>>,
    favorite: Option<bool>,
) -> Result<Value, String> {
    let client = reqwest::Client::new();
    let mut body = serde_json::Map::new();
    if let Some(t) = tags {
        body.insert("tags".into(), serde_json::to_value(t).unwrap());
    }
    if let Some(f) = favorite {
        body.insert("favorite".into(), serde_json::to_value(f).unwrap());
    }

    let resp = client
        .patch(format!("{}/genome/{}", SIDECAR_URL, genome_id))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Failed to connect: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Server error {}: {}", status, text));
    }

    resp.json::<Value>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}
