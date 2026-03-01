/**
 * Genome state management using Svelte 5 runes.
 */
import type { Genome, GenomeWithImage } from "../api";

export interface GenomeEntry {
  id: string;
  imageDataUrl: string;
  genome: Genome;
}

// Svelte 5 reactive state — use .value to read/write
class GenomeStore {
  entries: Map<string, GenomeEntry> = $state(new Map());
  selectedA: string | null = $state(null);
  selectedB: string | null = $state(null);

  get list(): GenomeEntry[] {
    return Array.from(this.entries.values());
  }

  get selectedParents(): [string | null, string | null] {
    return [this.selectedA, this.selectedB];
  }

  addFromResponse(responses: GenomeWithImage[]) {
    const next = new Map(this.entries);
    for (const r of responses) {
      next.set(r.id, {
        id: r.id,
        imageDataUrl: `data:image/png;base64,${r.image_base64}`,
        genome: r.genome,
      });
    }
    this.entries = next;
  }

  addSingle(r: GenomeWithImage) {
    const next = new Map(this.entries);
    next.set(r.id, {
      id: r.id,
      imageDataUrl: `data:image/png;base64,${r.image_base64}`,
      genome: r.genome,
    });
    this.entries = next;
  }

  remove(id: string) {
    const next = new Map(this.entries);
    next.delete(id);
    this.entries = next;
  }

  clear() {
    this.entries = new Map();
  }

  get(id: string): GenomeEntry | undefined {
    return this.entries.get(id);
  }

  updateGenome(id: string, updates: Partial<Genome>) {
    const entry = this.entries.get(id);
    if (entry) {
      const next = new Map(this.entries);
      next.set(id, {
        ...entry,
        genome: { ...entry.genome, ...updates },
      });
      this.entries = next;
    }
  }

  toggleSelect(id: string) {
    if (this.selectedA === id) {
      this.selectedA = null;
    } else if (this.selectedB === id) {
      this.selectedB = null;
    } else if (this.selectedA === null) {
      this.selectedA = id;
    } else if (this.selectedB === null) {
      this.selectedB = id;
    } else {
      this.selectedA = this.selectedB;
      this.selectedB = id;
    }
  }
}

export const store = new GenomeStore();
