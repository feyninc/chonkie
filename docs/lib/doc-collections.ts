import { docs } from "collections/server";

export function buildLoaderSourceInput() {
  return {
    docs: docs.toFumadocsSource(),
  } as const;
}
