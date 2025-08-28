import { SyntheticEvent, useState } from "react";
import NextImage, { ImageProps } from "next/image";
import { cn } from "@/lib/utils";

export default function Image(props: ImageProps) {
  const [ready, setReady] = useState(false);

  const handleLoad = (event: SyntheticEvent<HTMLImageElement, Event>) => {
    const img = event.target as HTMLImageElement;
    // Check if the image has actually loaded by verifying it has dimensions
    if (img.complete && img.naturalWidth > 0) {
      setReady(true);
    }
  };

  return <NextImage {...props} onLoad={handleLoad} className={cn("transition-opacity duration-300", ready ? "opacity-100" : "opacity-0", props.className)} />;
}
