// Throwaway measurement harness. Not part of the product.
import { StrictMode, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { FileTextIcon } from "lucide-react";
import { HelpCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import "./index.css";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetCloseButton,
} from "@/components/ui/sheet";

const params = new URLSearchParams(window.location.search);
const which = params.get("sheet") ?? "details";
const fontSize = Number(params.get("fs") ?? "15");
const previewWidth = Number(params.get("w") ?? "704");
const headerName = params.get("name") ?? "quarterly-report.pdf";
const pageParam = params.get("page");
const headerPage = pageParam === null ? 12 : Number(pageParam);

// Mirrors appearance-custom-store.ts: the preference drives --ui-font-scale
// only; the root font-size is never touched.
const UI_FONT_SIZE_CSS_BASE = 16;
const UI_FONT_SIZE_DEFAULT = 15;
if (fontSize !== UI_FONT_SIZE_DEFAULT) {
  document.documentElement.style.setProperty(
    "--ui-font-scale",
    String(fontSize / UI_FONT_SIZE_CSS_BASE),
  );
  document.documentElement.setAttribute("data-ui-font-size", String(fontSize));
}
document.documentElement.style.removeProperty("font-size");

function DetailsSheet() {
  return (
    <Sheet open={true}>
      <SheetContent
        side="right"
        className="w-[min(28rem,100vw)] p-0 sm:max-w-[28rem]"
        showCloseButton={false}
      >
        <SheetHeader className="border-b p-4">
          <div className="relative">
            <SheetTitle className="flex items-center gap-2 pr-10 font-heading text-base">
              <HugeiconsIcon
                icon={HelpCircleIcon}
                strokeWidth={1.75}
                className="size-icon text-chat-icon-fg"
              />
              Response details
            </SheetTitle>
            <SheetCloseButton className="absolute top-1/2 right-0 -translate-y-1/2" />
          </div>
          <SheetDescription className="sr-only">
            Timing, model, token, and tool details for this response.
          </SheetDescription>
        </SheetHeader>
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
          <button type="button">body-focusable</button>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function PreviewSheet() {
  return (
    <Sheet open={true}>
      <SheetContent
        side="right"
        style={{ width: previewWidth, maxWidth: "95vw" }}
        className="flex w-full flex-col gap-0 p-0"
        showCloseButton={false}
      >
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize document preview"
          className="absolute inset-y-0 left-0 z-20 w-2 cursor-col-resize transition-colors hover:bg-primary/25"
        />
        <SheetHeader className="gap-1 border-b p-4">
          <div className="relative">
            <SheetTitle className="flex items-center gap-2 pr-10 text-sm">
              <FileTextIcon className="size-4 shrink-0" />
              <span className="min-w-0 truncate">{headerName}</span>
              {headerPage != null && (
                <span className="shrink-0 text-muted-foreground">
                  &middot; page {headerPage}
                </span>
              )}
            </SheetTitle>
            <SheetCloseButton className="absolute top-1/2 right-0 -translate-y-1/2" />
          </div>
        </SheetHeader>
        <div className="min-h-0 flex-1">
          <button type="button">body-focusable</button>
        </div>
      </SheetContent>
    </Sheet>
  );
}

function Harness() {
  useEffect(() => {
    // Signal to the driver that React has committed.
    document.documentElement.setAttribute("data-harness-ready", "1");
  }, []);
  return which === "preview" ? <PreviewSheet /> : <DetailsSheet />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Harness />
  </StrictMode>,
);
