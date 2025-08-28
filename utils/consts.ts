// export const PLAYER_POS_Y = 1.6 + 0.1;
export const PLAYER_POS_Y = 1.598;
export const CAMERA_FOV = 65;

export const SEGMENT_COLORS = {
  wall: "#787878",
  door: "#780000",
  ceiling: "#510051",
  floor: "#787850",
  window: "#0080C0",
};

// export const SEGMENT_COLORS = {
//   wall: "#787878",
//   door: "#08FF33",
//   ceiling: "#787850",
//   floor: "#503232",
//   window: "#E6E6E6",
// };

export const SCENE_TYPES = ["Interior", "Exterior"] as const;
export const SPACE_TYPES = ["Residential Home", "Apartment/Condominium", "Luxury Villa"] as const;
export const ROOM_TYPES = ["Bathroom", "Bedroom", "Living Room", "Dining Room", "Hallway", "Kitchen", "Home Office"] as const;
export const STYLES = [
  "Modern",
  "Contemporary",
  "Colonial",
  "Art Deco",
  "Art Nouveau",
  "Asian Zen",
  "Baroque",
  "Bauhaus",
  "Boho",
  "Classic",
  "Coastal/Hamptons",
  "Eclectic",
  "Farmhouse",
  "French Country",
  "Gothic",
  "Historic",
  "Hollywood Regency",
  "Industrial",
  "Japandi",
  "Mediterranean",
  "Mid-Century",
  "Mid-Century Modern",
  "Minimalist",
  "Ornate",
  "Post-Industrial",
  "Rustic",
  "Scandinavian",
  "Shabby Chic",
  "Sleek Modern",
  "Southwestern",
  "Traditional",
  "Transitional",
  "Tropical",
  "Victorian",
] as const;
export const LIGHTING = ["Accent", "Ambient", "Fluorescent", "LED", "Natural", "Neon", "Task"] as const;
export const VIEWS = ["City Skyline", "Forest", "Garden", "Lakeside", "Mountain Range", "Oceanfront"] as const;

export const REF_IMAGES = [
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_1.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_2.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_3.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_4.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_5.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_6.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_7.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_8.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_9.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_10.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_11.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_12.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_13.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_14.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_15.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_16.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_17.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_18.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_19.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_20.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_21.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_22.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_23.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_24.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_25.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_26.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_27.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_28.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_29.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_30.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_31.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_32.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_33.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_34.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_35.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_36.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_37.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_38.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_39.webp",
  "https://res.cloudinary.com/dco2scymt/image/upload/v1755467184/steads/refs/interior_40.webp",
];

export const PROMPT = {
  render: {
    system:
      "Empty modern room interior, orange floor, grey structural walls, brown doors, blue glass windows with city skyscraper view when outside is visible, otherwise blue glass wardrobe doors when flat grey wall is behind. No furniture, clean architectural lines, preserved geometry, consistent camera angle and proportions",
    default: "Modern interior apartment with a minimalistic design",
    negative:
      "blurry, details are low, overlapping, grainy, multiple angles, deformed structures, unnatural, unrealistic, cartoon, anime, drawing, sketch, noise, jpeg artifacts, mutation, worst quality, normal quality, low quality, low res, messy, watermark, signature, cut off, low contrast, underexposed, overexposed, draft, disfigured, ugly, tiling, out of frame",
  },
  staging: {
    create: (options: string) => `
      You are an expert at staging, who given the image of the empty room identifies the dimensions, perspectives, windows, doors, exteriors of the windows, and leaves them intact. then uses the floor data to stage the place given the users prompts to fill it accordingly with the stuff user wants. you can also be creative and stage the room according to the interior design rules. You are an expert and have 50+ years of experience. mostly working with condos staging.

      MOST IMPORTANTLY you leave the structure of the room and you leave the perspective of the image and you leave the size of the image given intact, in your output image you create. You don't change any of these. same size image as given.


      Please stage this room with the options below. Maintain the picture size and dimensions size and where windows and doors are. have them all intact and just stage it. Stage the room fully and don't miss far spaces. make the staging as spacious as possible. do not make the staging limited at all.
      ${options}
    `,
  },
};
