/* -------------------------------------------------------------------------- */
/*                                easeInOutExpo                               */
/* -------------------------------------------------------------------------- */
export function easeInOutExpo(x: number) {
  return x === 0 ? 0 : x === 1 ? 1 : x < 0.5 ? Math.pow(2, 20 * x - 10) / 2 : (2 - Math.pow(2, -20 * x + 10)) / 2;
}

/* -------------------------------------------------------------------------- */
/*                                  easeInOut                                 */
/* -------------------------------------------------------------------------- */
export const easeInOut = (t: number) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t);

/* -------------------------------------------------------------------------- */
/*                                   zIndex                                   */
/* -------------------------------------------------------------------------- */
export const zIndex = (number: number = 1) => {
  return number * 0.001;
};

/* -------------------------------------------------------------------------- */
/*                                 sleep                                      */
/* -------------------------------------------------------------------------- */
export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/* -------------------------------------------------------------------------- */
/*                                 urlToBase64                                */
/* -------------------------------------------------------------------------- */
export async function urlToBase64(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const mimeType = response.headers.get("content-type") || "image/jpeg";

    return `data:${mimeType};base64,${buffer.toString("base64")}`;
  } catch (error) {
    console.error("Error converting URL to base64:", error);
    throw error;
  }
}

/* -------------------------------------------------------------------------- */
/*                        uploadBase64ImageToCloudinary                       */
/* -------------------------------------------------------------------------- */
export async function uploadBase64ImageToCloudinary({ base64Image, folder, preset }: { base64Image: string; folder: string; preset?: string }) {
  const base64Response = await fetch(base64Image);
  const blob = await base64Response.blob();

  const formData = new FormData();
  formData.append("file", blob, "image.jpg");
  formData.append("upload_preset", preset || process.env.NEXT_PUBLIC_CLOUDINARY_UPLOAD_PRESET!);
  formData.append("folder", folder);

  const response = await fetch(`https://api.cloudinary.com/v1_1/${process.env.NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME}/image/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) throw new Error("Upload failed");
  const result = await response.json();
  return result.secure_url as string;
}
