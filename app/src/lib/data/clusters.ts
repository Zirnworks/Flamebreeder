/**
 * Cluster definitions for the 30 conditional labels.
 * Labels from the Cluster Atlas — human-readable names for the
 * k-means clusters used in StyleGAN2-ADA conditional training.
 */

export interface ClusterDef {
  id: number;
  label: string;
  count: number;
  description: string;
  color: string; // CSS color for visual indicator
}

export const CLUSTERS: ClusterDef[] = [
  { id: 0,  label: "Violet Nebulae",     count: 3304, description: "Diffuse purple/magenta glows with wispy tendrils", color: "#9b59b6" },
  { id: 1,  label: "Ink Wisps",          count: 3565, description: "Thin calligraphic strokes in rainbow on black",    color: "#e74c3c" },
  { id: 2,  label: "Jeweled Tiles",      count: 2131, description: "Compact faceted symmetrical gem-like shapes",      color: "#1abc9c" },
  { id: 3,  label: "Filigree Mandalas",  count: 1481, description: "Ornate circular rosettes with lace-like detail",   color: "#f39c12" },
  { id: 4,  label: "Ember Blooms",       count: 2386, description: "Hot orange/red swirling fire-like forms",          color: "#e67e22" },
  { id: 5,  label: "Jade Mist",          count: 2611, description: "Green translucent bioluminescent fog",             color: "#27ae60" },
  { id: 6,  label: "Smoke Ribbons",      count: 2613, description: "Elongated sinuous colored smoke trails",           color: "#95a5a6" },
  { id: 7,  label: "Spark Clusters",     count: 2220, description: "Bright central bursts with radiating points",      color: "#f1c40f" },
  { id: 8,  label: "Arcane Sigils",      count: 2484, description: "Intricate symmetrical circuit/runic designs",      color: "#00bcd4" },
  { id: 9,  label: "Pale Phantoms",      count: 2075, description: "Near-monochrome ghostly low-saturation forms",     color: "#bdc3c7" },
  { id: 10, label: "Garden Tangles",     count: 2550, description: "Dense organic multi-colored floral tangles",       color: "#2ecc71" },
  { id: 11, label: "Dark Threads",       count: 4387, description: "Very sparse thin colored lines on black",          color: "#34495e" },
  { id: 12, label: "Petal Lanterns",     count: 2634, description: "Small isolated circular floral shapes",            color: "#8e44ad" },
  { id: 13, label: "Frost Stars",        count: 3190, description: "Radial snowflake/starburst crystalline patterns",  color: "#3498db" },
  { id: 14, label: "Spectral Webs",      count: 2482, description: "Diffuse web-like ethereal translucent layers",     color: "#d5a6e6" },
  { id: 15, label: "Golden Cruciform",   count: 3068, description: "Warm gold/amber cross-shaped ornamental forms",    color: "#d4a017" },
  { id: 16, label: "Prismatic Crosses",  count: 2246, description: "Strong axis lines with rainbow-colored rays",      color: "#e91e63" },
  { id: 17, label: "Neon Sprites",       count: 3057, description: "Bright angular spiky vivid energetic forms",       color: "#00e676" },
  { id: 18, label: "Arachnid Veils",     count: 3573, description: "Spidery tentacular forms with fine limbs",         color: "#5d4037" },
  { id: 19, label: "Crystal Blades",     count: 2179, description: "Sharp angular blade-like cool blue forms",         color: "#0097a7" },
  { id: 20, label: "Silver Thorns",      count: 3415, description: "Monochrome grey spiky urchin-like structures",     color: "#90a4ae" },
  { id: 21, label: "Amber Undergrowth",  count: 3206, description: "Dark olive/brown/gold earthy organic tangles",     color: "#795548" },
  { id: 22, label: "Neon Totems",        count: 3067, description: "Strong bilateral symmetry with vivid colors",      color: "#ff5722" },
  { id: 23, label: "Phantom Jellyfish",  count: 3528, description: "Translucent bulbous bioluminescent forms",         color: "#7c4dff" },
  { id: 24, label: "Midnight Blooms",    count: 2937, description: "Dense dark forms with deep blue/purple accents",   color: "#1a237e" },
  { id: 25, label: "Luminous Tendrils",  count: 3795, description: "Curling spiraling vine-like colored filaments",     color: "#ff9800" },
  { id: 26, label: "Cyan Plumes",        count: 2468, description: "Strongly blue/cyan feathery soft-edged forms",     color: "#00acc1" },
  { id: 27, label: "Pixel Lattices",     count: 1661, description: "Grid-aligned small repeating geometric motifs",    color: "#607d8b" },
  { id: 28, label: "Aqua Wisps",         count: 2677, description: "Thin sparse blue/teal strokes",                    color: "#26c6da" },
  { id: 29, label: "Chromatic Darts",    count: 2586, description: "Sharp fast-looking rainbow colored streaks",        color: "#ff1744" },
];

export const NUM_CLUSTERS = CLUSTERS.length;

/** Create a one-hot class label for a single cluster. */
export function oneHotLabel(clusterId: number): number[] {
  const label = new Array(NUM_CLUSTERS).fill(0);
  label[clusterId] = 1.0;
  return label;
}

/** Create a uniform (all-equal) class label. */
export function uniformLabel(): number[] {
  return new Array(NUM_CLUSTERS).fill(1.0 / NUM_CLUSTERS);
}

/** Normalize a label vector to sum to 1. */
export function normalizeLabel(label: number[]): number[] {
  const total = label.reduce((a, b) => a + b, 0);
  if (total <= 0) return uniformLabel();
  return label.map((v) => v / total);
}

/** Find the dominant cluster in a label vector. */
export function dominantCluster(label: number[] | null): ClusterDef | null {
  if (!label) return null;
  let maxIdx = 0;
  let maxVal = label[0];
  for (let i = 1; i < label.length; i++) {
    if (label[i] > maxVal) {
      maxVal = label[i];
      maxIdx = i;
    }
  }
  return CLUSTERS[maxIdx];
}
