import { generateOGImage } from "fumadocs-ui/og";

export const dynamic = "force-static";
export const alt = "Chonkie Documentation";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OGImage() {
  return generateOGImage({
    title: "Chonkie",
    description:
      "The lightweight ingestion library for fast, efficient and robust RAG pipelines",
    site: "Chonkie",
    primaryColor: "#a7896c",
    primaryTextColor: "#faf6e3",
  });
}
