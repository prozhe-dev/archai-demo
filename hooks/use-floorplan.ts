import { create } from "zustand";
import * as THREE from "three";
import { CAMERA_FOV, PLAYER_POS_Y } from "@/utils/consts";
import { Point, Shape } from "@/types/global";

export interface FloorplanData {
  doors: Shape[];
  floor: Point[];
  walls: Shape[];
  windows: Shape[];
  canvas: Point[];
}

export interface CameraState {
  mode: "pov" | "bev";
  state: "idle" | "transitioning";
  startTime: number;
  pov: {
    position: THREE.Vector3;
    rotation: THREE.Euler;
  };
  bev: {
    position: THREE.Vector3;
    rotation: THREE.Euler;
  };
}

export interface Bounds {
  min: THREE.Vector3;
  max: THREE.Vector3;
  center: THREE.Vector3;
}

export interface FloorPlanStore {
  data: FloorplanData | null;
  debug: boolean;
  bounds: Bounds;
  camera: CameraState;
  scene: {
    camera: null | THREE.Camera;
  };
  floorplanImage: string | null;
  setCamera: (camera: CameraState) => void;
  setData: (data: FloorplanData) => void;
  setBounds: (bounds: Bounds) => void;
  setDebug: (debug: boolean) => void;
  setScene: (scene: { camera: null | THREE.Camera }) => void;
  setFloorplanImage: (image: string) => void;
}

export const useFloorplan = create<FloorPlanStore>((set) => ({
  data: null,
  debug: false,
  bounds: {
    min: new THREE.Vector3(0, 0, 0),
    max: new THREE.Vector3(0, 0, 0),
    center: new THREE.Vector3(0, PLAYER_POS_Y, 0),
  },
  camera: {
    mode: "bev",
    state: "idle",
    startTime: 0,
    pov: {
      position: new THREE.Vector3(0, PLAYER_POS_Y, 0),
      rotation: new THREE.Euler(0, 0, 0),
    },
    bev: {
      position: new THREE.Vector3(0, 0, 0),
      rotation: new THREE.Euler(-Math.PI / 2, 0, 0),
    },
  },
  scene: {
    camera: null,
  },
  floorplanImage: null,
  setCamera: (camera: CameraState) => set({ camera }),
  setData: (data: FloorplanData) => {
    // Calculate bounds from floor data
    if (data.floor.length > 0) {
      const points = data.floor.map((p) => new THREE.Vector2(p[0], p[1]));
      const minX = Math.min(...points.map((p) => p.x));
      const maxX = Math.max(...points.map((p) => p.x));
      const minZ = Math.min(...points.map((p) => p.y));
      const maxZ = Math.max(...points.map((p) => p.y));

      const center = new THREE.Vector3((minX + maxX) / 2, PLAYER_POS_Y, (minZ + maxZ) / 2);

      const bounds = {
        min: new THREE.Vector3(minX, 0, minZ),
        max: new THREE.Vector3(maxX, 0, maxZ),
        center,
      };

      // Calculate optimal BEV height using the new function
      const optimalBEVHeight = calculateOptimalBEVHeight({ bounds });

      set((state) => ({
        data,
        camera: {
          ...state.camera,
          bev: {
            ...state.camera.bev,
            position: new THREE.Vector3(0, optimalBEVHeight, 0),
          },
        },
        bounds,
      }));
    } else {
      set({ data });
    }
  },
  setBounds: (bounds) => set({ bounds }),
  setDebug: (debug) => set({ debug }),
  setScene: (scene) => set({ scene }),
  setFloorplanImage: (image) => set({ floorplanImage: image }),
}));

interface CalculateOptimalBEVHeightProps {
  bounds: Bounds;
  margin?: number;
  fov?: number;
}
function calculateOptimalBEVHeight({ bounds, margin = 1.5, fov = CAMERA_FOV }: CalculateOptimalBEVHeightProps): number {
  const width = bounds.max.x - bounds.min.x;
  const depth = bounds.max.z - bounds.min.z;
  const maxDimension = Math.max(width, depth);

  // Convert FOV to radians
  const fovRadians = (fov * Math.PI) / 180;

  // Calculate the distance needed to fit the bounds within the camera's view
  // Using trigonometry: tan(fov/2) = opposite/adjacent
  // We want the opposite (half the max dimension) to fit within the camera's view
  const halfFov = fovRadians / 2;
  const distance = maxDimension / 2 / Math.tan(halfFov);

  // Apply margin
  const optimalHeight = distance * margin;

  return optimalHeight;
}
