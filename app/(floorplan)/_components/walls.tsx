import { useMemo } from "react";
import * as THREE from "three";
import { Edges, Extrude } from "@react-three/drei";
import { useFloorplan } from "@/hooks/use-floorplan";
import { RigidBody } from "@react-three/rapier";
import { zIndex } from "@/utils";
import { SEGMENT_COLORS } from "@/utils/consts";
import { DepthMaterial } from "@/components/materials/depth-material";

export default function Walls() {
  const { data } = useFloorplan();

  const shapes = useMemo(
    () =>
      data?.walls.map((wall) => {
        const shape = new THREE.Shape();

        // Move to first vertex
        shape.moveTo(wall[0][0][0], wall[0][0][1]);

        // Draw lines to other vertices
        for (let i = 1; i < wall.length; i++) {
          shape.lineTo(wall[i][0][0], wall[i][0][1]);
        }

        // Close the shape
        shape.lineTo(wall[0][0][0], wall[0][0][1]);

        return shape;
      }),
    [data],
  );

  return (
    <RigidBody type="fixed" colliders="trimesh">
      {/* Normal render (layer 0) */}
      <Extrude args={[shapes, { depth: 1, bevelEnabled: false }]} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color="white" />
        <Edges linewidth={2} color="black" threshold={45} />
      </Extrude>

      {/* Depth map (layer 1) */}
      <Extrude layers={[1]} args={[shapes, { depth: 1, bevelEnabled: false }]} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        {/* <meshDepthMaterial opacity={0.002} depthTest depthWrite /> */}
        <DepthMaterial />
      </Extrude>

      {/* Edge map (layer 2) */}
      <Extrude layers={[2]} args={[shapes, { depth: 1, bevelEnabled: false }]} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color="black" />
        <Edges linewidth={3} color="white" threshold={45} layers={[2]} />
      </Extrude>

      {/* Segment map (layer 3) */}
      <Extrude layers={[3]} args={[shapes, { depth: 1, bevelEnabled: false }]} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color={SEGMENT_COLORS.wall} />
        <Edges linewidth={2} color="black" threshold={45} layers={[3]} />
      </Extrude>
    </RigidBody>
  );
}
