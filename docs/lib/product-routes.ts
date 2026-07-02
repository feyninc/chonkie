import { getProductById } from "./docs-products";

export function getProductFromPath(pathname: string) {
  const path = pathname.replace(/\/$/, "") || "/";

  if (path.startsWith("/docs")) {
    return getProductById("chonkie");
  }

  if (path.startsWith("/python") || path.startsWith("/chonkie")) {
    return getProductById("chonkie");
  }

  return getProductById("chonkie");
}

export function switchProductHref(
  pathname: string,
  targetId: "chonkie" | "chonkiejs",
): string {
  const current = getProductFromPath(pathname);
  if (current.id === targetId) return pathname;

  const target = getProductById(targetId);
  if (target.externalUrl) return target.externalUrl;

  return target.defaultPage;
}
