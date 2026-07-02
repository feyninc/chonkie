import { CHONKIEJS_DOCS_URL } from "./constants";

export type DocsProductId = "chonkie" | "chonkiejs";

export interface DocsProduct {
  id: DocsProductId;
  label: string;
  description: string;
  basePath?: string;
  defaultPage: string;
  badge: string | null;
  /** When set, this product is hosted on a separate docs site. */
  externalUrl?: string;
}

export const docsProducts: DocsProduct[] = [
  {
    id: "chonkie",
    label: "chonkie",
    description: "chonkie on PyPI",
    basePath: "/chonkie",
    defaultPage: "/chonkie/quick-start",
    badge: null,
  },
  {
    id: "chonkiejs",
    label: "chonkiejs",
    description: "@chonkiejs/core on npm",
    defaultPage: CHONKIEJS_DOCS_URL,
    badge: "JS",
    externalUrl: CHONKIEJS_DOCS_URL,
  },
];

export function getProductById(id: DocsProductId): DocsProduct {
  const product = docsProducts.find((p) => p.id === id);
  if (!product) throw new Error(`Unknown product: ${id}`);
  return product;
}
