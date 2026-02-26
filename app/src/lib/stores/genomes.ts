/**
 * Genome state management using Svelte 5 runes.
 */
import { writable } from "svelte/store";
import type { Genome, GenomeWithImage } from "../api";

export interface GenomeEntry {
  id: string;
  imageDataUrl: string;
  genome: Genome;
}

function createGenomeStore() {
  const { subscribe, update, set } = writable<Map<string, GenomeEntry>>(
    new Map(),
  );

  return {
    subscribe,

    addFromResponse(responses: GenomeWithImage[]) {
      update((store) => {
        for (const r of responses) {
          store.set(r.id, {
            id: r.id,
            imageDataUrl: `data:image/png;base64,${r.image_base64}`,
            genome: r.genome,
          });
        }
        return new Map(store);
      });
    },

    addSingle(r: GenomeWithImage) {
      update((store) => {
        store.set(r.id, {
          id: r.id,
          imageDataUrl: `data:image/png;base64,${r.image_base64}`,
          genome: r.genome,
        });
        return new Map(store);
      });
    },

    remove(id: string) {
      update((store) => {
        store.delete(id);
        return new Map(store);
      });
    },

    clear() {
      set(new Map());
    },

    updateGenome(id: string, updates: Partial<Genome>) {
      update((store) => {
        const entry = store.get(id);
        if (entry) {
          store.set(id, {
            ...entry,
            genome: { ...entry.genome, ...updates },
          });
        }
        return new Map(store);
      });
    },
  };
}

export const genomes = createGenomeStore();
export const selectedParents = writable<[string | null, string | null]>([
  null,
  null,
]);
