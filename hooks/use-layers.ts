import { create } from "zustand";

export interface LayerStore {
  layer: number;
  setLayer: (layer: number) => void;
}

export const useLayer = create<LayerStore>((set, get) => ({
  layer: 0,
  setLayer: (layer: number) => set({ layer }),
}));
