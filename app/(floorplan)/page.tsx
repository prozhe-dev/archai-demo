"use client";

import { useFloorplan } from "@/hooks/use-floorplan";
import Scene from "./_components/scene";
import UploadFloorplan from "./_components/upload-floorplan";
import { AnimatePresence, motion } from "motion/react";

export default function Home() {
  const { data } = useFloorplan();
  return (
    <AnimatePresence mode="popLayout">
      {data ? (
        <motion.div key="scene" className="fixed inset-0 flex size-full flex-col" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3, delay: 0.5 }}>
          <Scene />
        </motion.div>
      ) : (
        <motion.div key="upload" className="fixed inset-0 flex size-full flex-col" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 2 }} transition={{ duration: 0.3 }}>
          <UploadFloorplan />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
