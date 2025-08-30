/* eslint-disable react-hooks/exhaustive-deps */
import * as THREE from "three";
import { useEffect, useRef } from "react";
import { RigidBody, RapierRigidBody, CapsuleCollider } from "@react-three/rapier";
import { useFrame, useThree } from "@react-three/fiber";
import { KeyboardControls, OrbitControls, useKeyboardControls } from "@react-three/drei";
import { Triplet } from "@/types/global";
import { useFloorplan } from "@/hooks/use-floorplan";
import { useHotkeys } from "react-hotkeys-hook";
import { easeInOut } from "@/utils";
import { useGenRender } from "@/hooks/use-gen-render";
import { useLayer } from "@/hooks/use-layers";

const direction = new THREE.Vector3();
const frontVector = new THREE.Vector3();
const sideVector = new THREE.Vector3();

interface FirstPersonDragControllerProps {
  target?: Triplet;
  playerMass?: number;
  initPos?: Triplet;
  moveSpeed?: number;
}

function FirstPersonDragController({ target = [0, 2.5, -3000] as Triplet, moveSpeed = 2.3, ...props }) {
  const DURATION = 2000;
  const { camera, setCamera, bounds, setScene } = useFloorplan();
  const { isGenRenderModalOpen } = useGenRender();
  const { layer } = useLayer();
  const state = useThree();

  const rigidRef = useRef<RapierRigidBody>(null);
  const [, get] = useKeyboardControls();

  useHotkeys("e", () => toggleCamera(), {
    enabled: layer === 0,
  });

  useEffect(() => {
    state.camera.layers.set(layer);
  }, [layer]);

  function toggleCamera() {
    if (camera.state === "transitioning") return;

    setCamera({
      ...camera,
      state: "transitioning",
      mode: camera.mode === "pov" ? "bev" : "pov",
      startTime: Date.now(),
      ...(camera.mode === "pov" && {
        pov: {
          position: state.camera.position.clone(),
          rotation: state.camera.rotation.clone(),
        },
      }),
    });
  }

  useEffect(() => {
    const timeout = setTimeout(() => {
      if (camera.mode === "bev") toggleCamera();
    }, 1000);

    return () => clearTimeout(timeout);
  }, []);

  useEffect(() => {
    if (state.camera) setScene({ camera: state.camera });
  }, [state.camera]);

  useFrame((state) => {
    if (camera.state === "transitioning") {
      const elapsed = Date.now() - camera.startTime;
      const progress = easeInOut(Math.min(elapsed / DURATION, 1));

      const targetCam = camera.mode === "pov" ? camera.pov : camera.bev;
      const fromCam = camera.mode === "pov" ? camera.bev : camera.pov;

      if (progress >= 1) {
        state.camera.position.copy(targetCam.position);
        state.camera.rotation.copy(targetCam.rotation);
        setCamera({ ...camera, state: "idle" });
      } else {
        state.camera.position.lerpVectors(fromCam.position, targetCam.position, progress);
        state.camera.rotation.x = THREE.MathUtils.lerp(fromCam.rotation.x, targetCam.rotation.x, progress);
        state.camera.rotation.y = THREE.MathUtils.lerp(fromCam.rotation.y, targetCam.rotation.y, progress);
        state.camera.rotation.z = THREE.MathUtils.lerp(fromCam.rotation.z, targetCam.rotation.z, progress);
      }
    } else if (camera.mode === "bev" && camera.state === "idle") {
      state.camera.position.copy(camera.bev.position);
      state.camera.rotation.copy(camera.bev.rotation);
    } else if (rigidRef.current && camera.mode === "pov" && camera.state === "idle") {
      const pos = rigidRef.current.translation();
      state.camera.position.set(pos.x, pos.y, pos.z);

      if (isGenRenderModalOpen) return;

      const { forward, backward, left, right } = get();
      const velocity = rigidRef.current.linvel();
      frontVector.set(0, 0, Number(backward) - Number(forward));
      sideVector.set(Number(left) - Number(right), 0, 0);
      direction.subVectors(frontVector, sideVector).normalize().multiplyScalar(moveSpeed).applyEuler(state.camera.rotation);

      rigidRef.current.setLinvel({ x: direction.x, y: velocity.y, z: direction.z }, true);
    }
  });

  return (
    <>
      <OrbitControls
        makeDefault
        target={target}
        enableZoom={false}
        enablePan={camera.mode === "pov" && camera.state === "idle"}
        enableRotate={false}
        // minPolarAngle={Math.PI / 2}
        // maxPolarAngle={Math.PI / 2}
        panSpeed={2}
        keyPanSpeed={0}
        dampingFactor={0.1}
        mouseButtons={{
          RIGHT: undefined,
          LEFT: THREE.MOUSE.PAN,
          MIDDLE: undefined,
        }}
      />
      <RigidBody ref={rigidRef} colliders={false} mass={1} type="dynamic" position={bounds.center} enabledRotations={[false, false, false]} {...props}>
        {/* <CapsuleCollider args={[0.45, 0.2]} /> */}
        <CapsuleCollider args={[0.45, 0.15]} />
        {/* <CuboidCollider args={[0.12, 0.6, 0.12]} /> */}
        {/* <CylinderCollider args={[0.1, 0.6]} /> */}
      </RigidBody>
    </>
  );
}

export default function Camera(props: FirstPersonDragControllerProps) {
  return (
    <KeyboardControls
      map={[
        { name: "forward", keys: ["ArrowUp", "w", "W"] },
        { name: "backward", keys: ["ArrowDown", "s", "S"] },
        { name: "left", keys: ["ArrowLeft", "a", "A"] },
        { name: "right", keys: ["ArrowRight", "d", "D"] },
      ]}
    >
      <FirstPersonDragController {...props} />
    </KeyboardControls>
  );
}
