import fs from "fs/promises";
import path from "path";
import { NextResponse } from "next/server";
import { FloorplanData } from "@/hooks/use-floorplan";

export async function GET() {
  const types = ["doors", "floor", "walls", "windows", "canvas"];

  const promises = types.map(async (type) => {
    const data = await fs.readFile(path.join(process.cwd(), "mockdata", "3", `${type}.txt`), "utf8");
    return {
      [type]: JSON.parse(data),
    };
  });

  const data = await Promise.all(promises);
  const res = data.reduce((acc, curr) => ({ ...acc, ...curr }), {}) as FloorplanData;
  return NextResponse.json(res);
}

export async function POST(req: Request) {
  const { image } = await req.json();
  if (!image) return NextResponse.json({ error: "No image provided" }, { status: 400 });

  const types = ["doors", "floor", "walls", "windows", "canvas"];

  const promises = types.map(async (type) => {
    const data = await fs.readFile(path.join(process.cwd(), "mockdata", "3", `${type}.txt`), "utf8");
    return {
      [type]: JSON.parse(data),
    };
  });

  const data = await Promise.all(promises);
  const res = data.reduce((acc, curr) => ({ ...acc, ...curr }), {}) as FloorplanData;
  return NextResponse.json(res);
}
