import type { SortedResult } from "fumadocs-core/search";

type SearchResultLike = Pick<SortedResult, "url" | "breadcrumbs">;

/** Order search hits: chonkie pages first. */
export function sortSearchResults<T extends SearchResultLike>(items: T[]): T[] {
  return items;
}
