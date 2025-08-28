"use client";

import { useFloorplan } from "@/hooks/use-floorplan";
import React, { Suspense, useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Grid, PerspectiveCamera, useEnvironment, View } from "@react-three/drei";
import { Physics } from "@react-three/rapier";
import Camera from "@/app/(floorplan)/_components/camera";
import Ceiling from "@/app/(floorplan)/_components/ceiling";
import Floor from "@/app/(floorplan)/_components/floor";
import Walls from "@/app/(floorplan)/_components/walls";
import Doors from "@/app/(floorplan)/_components/doors";
import { useHotkeys } from "react-hotkeys-hook";
import * as THREE from "three";
import { CAMERA_FOV } from "@/utils/consts";
import FloorPlanImage from "./floorplan-image";
import SnapshotTaker from "./snapshot-taker";
import GenRender from "./gen-render";
import Windows from "./windows";
import Minimap from "./minimap";
import { useSpring } from "@react-spring/three";
import { easeInOut } from "@/utils";
import { useLayer } from "@/hooks/use-layers";

export default function Scene() {
  const { data, debug, setDebug, camera, bounds } = useFloorplan();
  const { layer, setLayer } = useLayer();
  const containerRef = useRef<HTMLDivElement>(null);
  useHotkeys("k", () => setDebug(!debug));

  useHotkeys("0", () => setLayer(0));
  useHotkeys("1", () => setLayer(1));
  useHotkeys("2", () => setLayer(2));
  useHotkeys("3", () => setLayer(3));

  if (!data) {
    return <div className="flex size-full items-center justify-center">Data not found.</div>;
  }

  return (
    <div ref={containerRef} className="relative h-screen w-full">
      {debug && (
        <span className="absolute bottom-0 left-1/2 z-10 m-4 -translate-x-1/2 text-red-400">
          {camera.mode} - {camera.state}
        </span>
      )}
      <View className="absolute inset-0">
        {layer === 0 ? <FadableEnvironment files="/images/sunset_jhbcentral_1k.exr" show={camera.mode === "pov" && camera.state === "idle"} bgFallback="white" /> : <color attach="background" args={["black"]} />}
        <PerspectiveCamera makeDefault fov={CAMERA_FOV} />

        <group position={[-bounds.center.x, 0, -bounds.center.z]}>
          <Physics debug={debug}>
            <Floor />
            <Camera />
            <Walls />
            <Windows />
            <Doors />
            <Ceiling />
            <FloorPlanImage />
          </Physics>
        </group>

        {layer === 0 && <SnapshotTaker />}
        {(debug || camera.mode === "bev") && (
          <Grid position={[0, 0, 0]} args={[20, 20]} cellSize={1} cellThickness={0.5} cellColor="#cccccc" sectionSize={5} sectionThickness={1} sectionColor="#999999" infiniteGrid={true} />
        )}
      </View>

      <View
        className="fixed bottom-0 left-0 z-10 m-5 size-[20rem] origin-bottom-left scale-50 cursor-pointer opacity-0 transition-all duration-300 hover:scale-105 data-[active=true]:opacity-100"
        style={{ border: "2px solid grey" }}
        data-active={camera.mode === "pov" && camera.state === "idle"}
        visible={camera.mode === "pov" && camera.state === "idle"}
      >
        <Minimap />
      </View>

      <Canvas gl={{ antialias: true, powerPreference: "high-performance" }} shadows={true} className="absolute inset-0" eventSource={containerRef as any}>
        <Suspense fallback={null}>
          <View.Port />
        </Suspense>
      </Canvas>

      <GenRender />
    </div>
  );
}

function FadableEnvironment({ files, show, bgFallback = "white" }: { files: string | string[]; show: boolean; bgFallback?: THREE.ColorRepresentation }) {
  const { scene } = useThree();
  const envTex = useEnvironment({ files });

  // Mount the env as scene.environment once
  useEffect(() => {
    const prevEnv = scene.environment;
    scene.environment = envTex;
    return () => {
      scene.environment = prevEnv;
    };
  }, [envTex, scene]);

  // Ensure we have a fallback background color when faded out
  useEffect(() => {
    if (!scene.background || scene.background === envTex) {
      scene.background = new THREE.Color(bgFallback);
    }
  }, [scene, envTex, bgFallback]);

  const TO_INTENSITY = 50;

  const { t } = useSpring({
    t: show ? 1 : TO_INTENSITY,
    config: { duration: 200, easing: (t) => easeInOut(t) },
  });

  useFrame(() => {
    const k = t.get();

    if (k < TO_INTENSITY - 0.0001) {
      if (scene.background !== envTex) scene.background = envTex as any;
      scene.backgroundIntensity = k;
    } else {
      if (!(scene.background instanceof THREE.Color)) {
        scene.background = new THREE.Color(bgFallback);
      }
      scene.backgroundIntensity = TO_INTENSITY;
    }
  });

  return null;
}
