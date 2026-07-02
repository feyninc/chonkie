"use client";

import { Check } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  docsProducts,
  type DocsProduct,
} from "@/lib/docs-products";
import {
  getProductFromPath,
  switchProductHref,
} from "@/lib/product-routes";

const CHONKIE_ICON = "/assets/logo/chonkie_logo_br_transparent_bg.png";

function ProductIcon({ badge }: { badge: string | null }) {
  if (!badge) {
    return (
      <img src={CHONKIE_ICON} alt="" width={18} height={18} className="shrink-0 object-contain" />
    );
  }

  return (
    <div className="relative shrink-0">
      <img src={CHONKIE_ICON} alt="" width={18} height={18} className="object-contain" />
      <span className="absolute -bottom-0.5 -right-1 text-[7px] font-bold leading-none bg-fd-primary text-fd-primary-foreground rounded px-0.5">
        {badge}
      </span>
    </div>
  );
}

function ProductBadge({ badge }: { badge: string }) {
  return (
    <span className="shrink-0 border border-dashed border-fd-foreground/25 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-fd-muted-foreground">
      {badge}
    </span>
  );
}

export function ProductSwitcher() {
  const pathname = usePathname() ?? "/";
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const current = getProductFromPath(pathname);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    if (open) document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, [open]);

  function handleSelect(product: DocsProduct) {
    setOpen(false);
    if (product.id === current.id) return;

    const href = switchProductHref(pathname, product.id);
    if (product.externalUrl) {
      window.location.assign(href);
      return;
    }

    router.push(href);
  }

  return (
    <div
      ref={containerRef}
      data-product-switcher
      className="relative -mx-4 border-y border-fd-border"
    >
      <button
        type="button"
        data-product-switcher-trigger
        onClick={() => setOpen((v) => !v)}
        className="group flex w-full items-center gap-2.5 px-4 py-2.5 text-sm text-fd-muted-foreground transition-colors hover:bg-fd-foreground/[0.03] hover:text-fd-foreground"
      >
        <ProductIcon badge={current.badge} />
        <span className="truncate font-medium">{current.label}</span>
        {current.badge && <ProductBadge badge={current.badge} />}
        <svg
          className={`ml-auto size-3.5 shrink-0 text-fd-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}
          viewBox="0 0 10 6"
          fill="none"
          aria-hidden
        >
          <path d="M1 1L5 5L9 1" stroke="currentColor" strokeWidth="1.2" />
        </svg>
      </button>

      {open && (
        <div className="absolute top-full left-0 right-0 z-50 border border-fd-border bg-fd-popover py-1 shadow-lg shadow-fd-foreground/5">
          {docsProducts.map((product) => {
            const isActive = product.id === current.id;
            return (
              <button
                key={product.id}
                type="button"
                onClick={() => handleSelect(product)}
                className="flex w-full items-center gap-2.5 px-4 py-2 text-start text-sm transition-colors hover:bg-fd-foreground/[0.04]"
              >
                <span className="flex w-4 shrink-0 justify-center">
                  {isActive && (
                    <Check className="size-3.5 text-fd-primary" strokeWidth={2.5} />
                  )}
                </span>
                <ProductIcon badge={product.badge} />
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="font-medium text-fd-foreground">
                    {product.label}
                  </span>
                  <span className="truncate text-xs text-fd-muted-foreground">
                    {product.description}
                  </span>
                </span>
                {product.badge && <ProductBadge badge={product.badge} />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
