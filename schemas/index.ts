import { PROMPT, SCENE_TYPES } from "@/utils/consts";
import { z } from "zod";

export const GenRenderFormOptionsSchema = z.object({
  sceneType: z.enum(SCENE_TYPES),
  spaceType: z.string().optional(),
  roomType: z.string().optional(),
  style: z.string().optional(),
  lighting: z.string().optional(),
  view: z.string().optional(),
});

export const GenRenderFormSchema = GenRenderFormOptionsSchema.extend({
  image: z.string().min(1, "Image is required"),
  referenceImage: z.string().min(1, "Reference image is required"),
  prompt: z.string().max(32000, "Prompt must be less than 32000 characters").optional(),
});

export const GenRenderSchema = z.object({
  image: z.string().min(1, "Image is required"),
  passes: z.object({
    depth: z.string().min(1, "Depth Pass is required"),
    edges: z.string().min(1, "Edges Pass is required"),
    segments: z.string().min(1, "Segments Pass is required"),
  }),
  referenceImage: z.string().min(1, "Reference image is required"),
  negativePrompt: z.string().default(PROMPT.render.negative),
  prompt: z.string(),
  outputFormat: z.enum(["webp", "jpg", "png"]).default("jpg"),
  styleTransferStrength: z.number().min(0).max(1).default(1),
  base64_response: z.boolean().default(false),
});

export const GenStagedRenderSchema = z.object({
  image: z.string().min(1, "Image is required"),
  options: z.string().min(1, "Options are required"),
});
