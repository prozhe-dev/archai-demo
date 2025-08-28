import { NextResponse, NextRequest } from "next/server";
import { GenRenderSchema } from "@/schemas";
import { urlToBase64 } from "@/utils";

export async function POST(req: NextRequest) {
  try {
    const host = process.env.RENDER_API_HOST;
    const key = process.env.RENDER_API_KEY;
    const endpoint = process.env.RENDER_API_ENDPOINT;
    if (!host || !key || !endpoint) throw new Error("API is missing required environment variables");

    const body = await req.json();
    const input = GenRenderSchema.parse(body);
    if (!input) throw new Error("Invalid input");

    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        "x-rapidapi-key": key,
        "x-rapidapi-host": host,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(input),
    });

    if (!res.ok) throw new Error("Failed to get render");

    const data = (await res.json()) as { output: string };

    const render = await urlToBase64(data.output);

    return NextResponse.json({ render });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Failed to get base render" }, { status: 500 });
  }
}
