import { useMemo } from "react";
import * as THREE from "three";
import { Edges, Extrude } from "@react-three/drei";
import { useFloorplan } from "@/hooks/use-floorplan";
import { RigidBody } from "@react-three/rapier";
import { zIndex } from "@/utils";
import { SEGMENT_COLORS } from "@/utils/consts";

export default function Windows() {
  const { data } = useFloorplan();

  const shapes = useMemo(
    () =>
      data?.windows.map((window) => {
        const shape = new THREE.Shape();

        // Move to first vertex
        shape.moveTo(window[0][0][0], window[0][0][1]);

        // Draw lines to other vertices
        for (let i = 1; i < window.length; i++) {
          shape.lineTo(window[i][0][0], window[i][0][1]);
        }

        // Close the shape
        shape.lineTo(window[0][0][0], window[0][0][1]);

        return shape;
      }),
    [data],
  );

  return (
    <RigidBody type="fixed" colliders="trimesh">
      {/* Normal render (layer 0) */}
      <Extrude renderOrder={1} args={[shapes, { depth: 1, bevelEnabled: false }]} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color="#6ca8ff" transparent opacity={0.5} side={THREE.BackSide} />
        <Edges linewidth={2} color="black" threshold={45} />
      </Extrude>

      {/* Depth map (layer 1) */}

      {/* Edge map (layer 2) */}
      <Extrude layers={[2]} args={[shapes, { depth: 1, bevelEnabled: false }]} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color="black" transparent opacity={0.2} />
        <Edges linewidth={3} color="white" threshold={45} layers={[2]} />
      </Extrude>

      {/* Segment map (layer 3) */}
      <Extrude layers={[3]} args={[shapes, { depth: 1, bevelEnabled: false }]} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color={SEGMENT_COLORS.window} />
        <Edges linewidth={2} color="black" threshold={45} layers={[3]} />
      </Extrude>
    </RigidBody>
  );
}
