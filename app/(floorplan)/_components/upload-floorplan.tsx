"use client";

import React, { useEffect } from "react";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import Dropzone from "react-dropzone";
import { useFloorplan } from "@/hooks/use-floorplan";
import { Form, FormControl, FormField, FormItem, FormMessage } from "@/components/ui/form";
import { cn } from "@/lib/utils";
import { ImageIcon } from "lucide-react";
import { toast } from "sonner";
import { uploadBase64ImageToCloudinary } from "@/utils";

const UploadSchema = z.object({
  image: z.string().min(1, "Image is required"),
});

export default function UploadFloorplan() {
  const { setData, setFloorplanImage } = useFloorplan();

  const form = useForm<z.infer<typeof UploadSchema>>({
    resolver: zodResolver(UploadSchema),
    mode: "onChange",
    defaultValues: { image: "" },
  });

  const {
    formState: { isSubmitting, isValid, isLoading },
    handleSubmit,
    setError,
    watch,
  } = form;

  const image = watch("image");

  async function onSubmit(formData: z.infer<typeof UploadSchema>) {
    console.log({ formData });
    try {
      const url = await uploadBase64ImageToCloudinary({ base64Image: formData.image, folder: "steads/floorplans" });
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/vertx`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: process.env.FLOORPLAN_DETECTOR_API_KEY!,
        },
        body: JSON.stringify({ image: formData.image }),
      });
      if (!res.ok) throw new Error("Failed to upload floorplan");
      const data = await res.json();
      console.log({ data });
      setData(data);
      setFloorplanImage(url);
    } catch (err) {
      toast.error("Upload failed", { description: err instanceof Error ? err.message : "Something went wrong" });
    }
  }

  useEffect(() => {
    if (image) handleSubmit(onSubmit)();
  }, [image]);

  return (
    <div className="flex size-full flex-col items-center justify-center">
      <Form {...form}>
        <form className={cn("flex w-full max-w-xl flex-col gap-6", isSubmitting && "animate-pulse")}>
          <FormField
            control={form.control}
            name="image"
            render={({ field }) => (
              <FormItem>
                <FormControl>
                  <div className="w-full overflow-hidden">
                    <Dropzone
                      multiple={false}
                      accept={{ "image/*": [".jpeg", ".jpg", ".png", ".gif", ".webp"] }}
                      onDrop={(acceptedFiles) => {
                        const file = acceptedFiles[0];
                        if (file) {
                          const reader = new FileReader();
                          reader.onload = (e) => {
                            const base64 = e.target?.result as string;
                            field.onChange(base64);
                          };
                          reader.readAsDataURL(file);
                        }
                      }}
                      onDropRejected={(errors) => {
                        setError("image", { message: errors[0].errors[0].message });
                      }}
                    >
                      {({ getRootProps, getInputProps, isDragActive }) => (
                        <div
                          {...getRootProps()}
                          className={cn(
                            "group/dropzone relative flex w-full cursor-pointer flex-col overflow-hidden rounded-md border transition-all",
                            isDragActive ? "bg-secondary/50 opacity-50" : "hover:bg-secondary",
                            field.value ? "border-secondary" : "aspect-video border-dashed",
                          )}
                        >
                          <input {...getInputProps()} />

                          {field.value && (
                            <div className="bg-secondary relative w-full overflow-hidden rounded-md border transition-all group-hover/dropzone:opacity-50">
                              <img src={field.value} alt="Floorplan preview" className="w-full object-contain" />
                            </div>
                          )}

                          <div className={cn("absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center transition-all", field.value && "opacity-0 group-hover/dropzone:opacity-100")}>
                            <ImageIcon className="size-8" />
                            <p className="text-sm">{isDragActive ? "Drop your floorplan here" : "Drag & drop your floorplan here, or click to select"}</p>
                          </div>
                        </div>
                      )}
                    </Dropzone>
                  </div>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </form>
      </Form>
    </div>
  );
}
