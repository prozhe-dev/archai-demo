import { NextResponse, NextRequest } from "next/server";
import { GenStagedRenderSchema } from "@/schemas";
import OpenAI, { toFile } from "openai";
import { PROMPT } from "@/utils/consts";

const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) throw Error("OPENAI_API_KEY is not set");
const openai = new OpenAI({ apiKey });

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const input = GenStagedRenderSchema.parse(body);
    if (!input) throw new Error("Invalid input");

    const { image: imageUrl, options } = input;

    const image = await imageToFile(imageUrl);
    const prompt = PROMPT.staging.create(options);

    const response = await openai.images.edit({
      model: "gpt-image-1",
      image,
      prompt,
      size: "1536x1024",
      quality: "high",
      input_fidelity: "high",
      n: 1,
    });

    if (!response.data || response.data.length === 0) {
      throw new Error("No image data received from OpenAI API");
    }

    const render = `data:image/png;base64,${response.data[0].b64_json!}`;

    return NextResponse.json({ render });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Failed to get staged render" }, { status: 500 });
  }
}

async function imageToFile(image: string) {
  const response = await fetch(image);
  const blob = await response.blob();
  const file = await toFile(blob, `image-${Date.now()}.png`, { type: blob.type || "image/png" });
  return file;
}
