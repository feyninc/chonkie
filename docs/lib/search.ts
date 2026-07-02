import { createFromSource } from "fumadocs-core/search/server";
import { findPath } from "fumadocs-core/page-tree";
import { source } from "@/lib/source";

type DocsPage = ReturnType<typeof source.getPages>[number];

async function resolveStructuredData(page: DocsPage) {
  const data = page.data as {
    structuredData?: unknown | (() => unknown | Promise<unknown>);
    load?: () => Promise<{ structuredData?: unknown }>;
  };

  if (data.structuredData) {
    return typeof data.structuredData === "function"
      ? await data.structuredData()
      : data.structuredData;
  }

  if (typeof data.load === "function") {
    return (await data.load()).structuredData;
  }

  return undefined;
}

function resolveTreeBreadcrumbs(page: DocsPage): string[] {
  const breadcrumbs: string[] = [];
  const pageTree = source.getPageTree(page.locale);
  const path = findPath(
    pageTree.children,
    (node) => node.type === "page" && node.url === page.url,
  );

  if (!path) return breadcrumbs;

  path.pop();
  if (pageTree.name) breadcrumbs.push(String(pageTree.name));

  for (const segment of path) {
    if (segment.name) breadcrumbs.push(String(segment.name));
  }

  return breadcrumbs;
}

async function buildPageSearchIndex(page: DocsPage) {
  const structuredData = await resolveStructuredData(page);
  if (!structuredData) {
    throw new Error(
      `Cannot index "${page.url}": missing structured data. Ensure the page is valid MDX.`,
    );
  }

  const treeBreadcrumbs = resolveTreeBreadcrumbs(page);
  const breadcrumbs = ["chonkie", ...treeBreadcrumbs.filter((crumb) => crumb !== "chonkie")];

  return {
    id: page.url,
    title: page.data.title ?? page.url,
    description: page.data.description,
    url: page.url,
    breadcrumbs: breadcrumbs.length > 0 ? breadcrumbs : undefined,
    structuredData,
  };
}

export const searchServer = createFromSource(source, {
  buildIndex: buildPageSearchIndex,
  search: {
    limit: 24,
    groupBy: {
      properties: ["page_id"],
      maxResult: 12,
    },
  },
});
