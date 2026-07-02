import { createMDX } from "fumadocs-mdx/next";

const withMDX = createMDX();

const basePath = process.env.BASE_PATH || "";

/** @type {import('next').NextConfig} */
const config = {
  output: "export",
  basePath,
  trailingSlash: true,
  reactStrictMode: true,
  images: {
    unoptimized: true,
    remotePatterns: [{ hostname: "raw.githubusercontent.com" }],
  },
};

export default withMDX(config);
