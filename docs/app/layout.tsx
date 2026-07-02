import { RootProvider } from "fumadocs-ui/provider/next";
import { Lora } from "next/font/google";
import type { ReactNode } from "react";
import type { Metadata } from "next";
import { ChonkieSearchDialog } from "@/components/search-dialog";
import { DOCS_SITE_URL } from "@/lib/constants";
import "./global.css";

const lora = Lora({
  subsets: ["latin"],
  variable: "--font-lora",
});

export const metadata: Metadata = {
  metadataBase: new URL(DOCS_SITE_URL),
  title: "Chonkie Documentation",
  description:
    "The lightweight ingestion library for fast, efficient and robust RAG pipelines",
  icons: {
    icon: "/assets/logo/chonkie_logo_br_transparent_bg.png",
  },
  openGraph: {
    title: "Chonkie Documentation",
    description:
      "The lightweight ingestion library for fast, efficient and robust RAG pipelines",
    siteName: "Chonkie",
  },
  twitter: {
    card: "summary_large_image",
  },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${lora.className} ${lora.variable}`}
        suppressHydrationWarning
      >
        <RootProvider
          search={{
            SearchDialog: ChonkieSearchDialog,
            options: { type: "static" },
          }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
