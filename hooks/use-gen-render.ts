import { create } from "zustand";

export interface GenRenderStore {
  isGenRenderModalOpen: boolean;
  snapshotImage: string | null;
  snapshotDepthImage: string | null;
  snapshotEdgesImage: string | null;
  snapshotSegmentsImage: string | null;
  renderImage: string | null;
  stagedImage: string | null;
  setSnapshotImage: (image: string | null) => void;
  setSnapshotDepthImage: (image: string | null) => void;
  setSnapshotEdgesImage: (image: string | null) => void;
  setSnapshotSegmentsImage: (image: string | null) => void;
  setGenRenderModalOpen: (isOpen: boolean) => void;
  setRenderImage: (image: string | null) => void;
  setStagedImage: (image: string | null) => void;
  resetImages: () => void;
}

export const useGenRender = create<GenRenderStore>((set, get) => ({
  isGenRenderModalOpen: false,
  snapshotImage: null,
  snapshotDepthImage: null,
  snapshotEdgesImage: null,
  snapshotSegmentsImage: null,
  renderImage: null,
  stagedImage: null,
  setSnapshotImage: (image: string | null) => set({ snapshotImage: image }),
  setSnapshotDepthImage: (image: string | null) => set({ snapshotDepthImage: image }),
  setSnapshotEdgesImage: (image: string | null) => set({ snapshotEdgesImage: image }),
  setSnapshotSegmentsImage: (image: string | null) => set({ snapshotSegmentsImage: image }),
  setGenRenderModalOpen: (isOpen: boolean) => set({ isGenRenderModalOpen: isOpen }),
  setRenderImage: (image: string | null) => set({ renderImage: image }),
  setStagedImage: (image: string | null) => set({ stagedImage: image }),
  resetImages: () => set({ snapshotImage: null, renderImage: null, stagedImage: null }),
}));
