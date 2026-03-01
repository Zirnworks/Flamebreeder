/**
 * API layer — wraps Tauri IPC commands for the inference server.
 */
import { invoke } from "@tauri-apps/api/core";

export interface Genome {
  id: string;
  latent_vector: number[];
  latent_space: string;
  truncation_psi: number | null;
  class_label: number[] | null;
  seed_z: number[] | null;
  parents: [string, string] | null;
  breeding_method: string | null;
  breeding_params: Record<string, unknown> | null;
  generation: number;
  created_at: string;
  tags: string[];
  favorite: boolean;
}

export interface GenomeWithImage {
  id: string;
  image_base64: string;
  genome: Genome;
}

export async function generateRandom(
  count: number,
  truncationPsi: number = 0.7,
  classLabel?: number[],
): Promise<GenomeWithImage[]> {
  return invoke("generate_random", { count, truncationPsi, classLabel });
}

export async function breed(
  parentAId: string,
  parentBId: string,
  method: string = "average",
  params: Record<string, unknown> = {},
  count: number = 1,
): Promise<GenomeWithImage[]> {
  return invoke("breed", {
    parentAId,
    parentBId,
    method,
    params,
    count,
  });
}

export async function interpolate(
  genomeAId: string,
  genomeBId: string,
  steps: number = 10,
  method: string = "slerp",
): Promise<GenomeWithImage[]> {
  return invoke("interpolate", { genomeAId, genomeBId, steps, method });
}

export async function mutateGenome(
  genomeId: string,
  rate: number = 0.1,
  strength: number = 0.3,
): Promise<GenomeWithImage> {
  return invoke("mutate_genome", { genomeId, rate, strength });
}

export async function remapGenome(
  genomeId: string,
  classLabel: number[],
  truncationPsi: number = 0.7,
): Promise<GenomeWithImage> {
  return invoke("remap_genome", { genomeId, classLabel, truncationPsi });
}

export async function getGenome(genomeId: string): Promise<GenomeWithImage> {
  return invoke("get_genome", { genomeId });
}

export async function listGenomes(): Promise<Genome[]> {
  return invoke("list_genomes");
}

export async function checkServerHealth(): Promise<Record<string, unknown>> {
  return invoke("check_server_health");
}

export async function updateGenome(
  genomeId: string,
  tags?: string[],
  favorite?: boolean,
): Promise<Genome> {
  return invoke("update_genome", { genomeId, tags, favorite });
}
