import type * as PageTree from "fumadocs-core/page-tree";

function folderSlug(folder: PageTree.Folder): string | null {
  const url =
    folder.index?.url ??
    folder.children.find((c) => c.type === "page")?.url;

  if (!url || typeof url !== "string") return null;

  const parts = url.split("/").filter(Boolean);
  return parts[0] ?? null;
}

function findProductFolder(
  nodes: PageTree.Node[],
  slug: string,
): PageTree.Folder | null {
  for (const node of nodes) {
    if (node.type !== "folder") continue;
    if (folderSlug(node) === slug) return node;
    const nested = findProductFolder(node.children, slug);
    if (nested) return nested;
  }
  return null;
}

/** Show only the chonkie sidebar pages (chonkiejs is hosted separately). */
export function filterPageTreeForProduct(tree: PageTree.Root): PageTree.Root {
  const folder =
    findProductFolder(tree.children, "chonkie") ??
    (tree.fallback ? findProductFolder(tree.fallback.children, "chonkie") : null);

  if (!folder) return tree;

  return {
    ...tree,
    children: folder.children,
  };
}
