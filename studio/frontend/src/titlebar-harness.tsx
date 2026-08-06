// Throwaway harness for validating unslothai/unsloth#7957 on real Windows and
// macOS runners. Mirrors app/provider.tsx's desktop branch (the custom-titlebar
// return at the end of TauriWrapper) and mounts the real WindowTitlebar next to
// each sheet, so the window-control band is rendered exactly as the desktop app
// renders it.
//
// ?variant=before  default close button, `absolute top-4 right-4` (main)
// ?variant=after   PR #7957 as merged: centred on the title row
// ?variant=fixed   PR #7957 plus the titlebar offset
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import type { CSSProperties } from "react";
import { FileTextIcon } from "lucide-react";
import { HelpCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

import "./index.css";
import {
  WindowTitlebar,
  shouldUseCustomWindowTitlebar,
  shouldUseNativeMacWindowTitlebar,
} from "@/components/tauri/window-titlebar";
import { DesktopChromeVarsEffect } from "@/app/provider";
import {
  Sheet,
  SheetCloseButton,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

const params = new URLSearchParams(window.location.search);
const variant = params.get("variant") ?? "after";
const which = params.get("sheet") ?? "details";
const fontSize = Number(params.get("fs") ?? "15");

if (fontSize !== 15) {
  document.documentElement.style.setProperty(
    "--ui-font-scale",
    String(fontSize / 16),
  );
  document.documentElement.setAttribute("data-ui-font-size", String(fontSize));
}
document.documentElement.style.removeProperty("font-size");

// Verbatim copy of CUSTOM_CHROME_STYLE from app/provider.tsx.
const CUSTOM_CHROME_STYLE = {
  "--studio-titlebar-height": "0px",
  "--studio-custom-titlebar-height": "34px",
  "--studio-desktop-titlebar-height": "34px",
  "--studio-sidebar-expanded-width": "17.5rem",
  "--studio-sidebar-collapsed-width": "3rem",
  "--studio-collapsed-chat-controls-inset": "12px",
  "--studio-startup-top-inset": "42px",
  "--studio-content-top-inset": "34px",
  "--studio-hidden-route-top-inset": "34px",
  "--studio-chat-header-height": "48px",
  "--studio-chat-header-padding-top": "9px",
  "--studio-media-header-left-inset": "0.5rem",
  "--studio-chat-control-height": "33px",
  "--studio-chat-header-right-inset": "0px",
  "--studio-window-control-inset": "112px",
} as CSSProperties;

// Only the `fixed` variant carries the titlebar offset.
const offsetStyle: CSSProperties =
  variant === "fixed"
    ? {
        height: "calc(100% - var(--studio-custom-titlebar-height, 0px))",
        marginTop: "var(--studio-custom-titlebar-height, 0px)",
      }
    : {};

function DetailsSheet() {
  const centred = variant !== "before";
  return (
    <Sheet open={true}>
      <SheetContent
        side="right"
        className="w-[min(28rem,100vw)] p-0 sm:max-w-[28rem]"
        showCloseButton={!centred}
        style={offsetStyle}
      >
        <SheetHeader className="border-b p-4">
          {centred ? (
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
          ) : (
            <SheetTitle className="flex items-center gap-2 pr-10 font-heading text-base">
              <HugeiconsIcon
                icon={HelpCircleIcon}
                strokeWidth={1.75}
                className="size-icon text-chat-icon-fg"
              />
              Response details
            </SheetTitle>
          )}
          <SheetDescription className="sr-only">
            Timing, model, token, and tool details for this response.
          </SheetDescription>
        </SheetHeader>
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4" />
      </SheetContent>
    </Sheet>
  );
}

function PreviewSheet() {
  const centred = variant !== "before";
  const title = (
    <SheetTitle className="flex items-center gap-2 pr-10 text-sm">
      <FileTextIcon className="size-4 shrink-0" />
      <span className="min-w-0 truncate">quarterly-report.pdf</span>
      <span className="shrink-0 text-muted-foreground">&middot; page 12</span>
    </SheetTitle>
  );
  return (
    <Sheet open={true}>
      <SheetContent
        side="right"
        style={{ width: 704, maxWidth: "95vw", ...offsetStyle }}
        className="flex w-full flex-col gap-0 p-0"
        showCloseButton={!centred}
      >
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize document preview"
          className="absolute inset-y-0 left-0 z-20 w-2 cursor-col-resize transition-colors hover:bg-primary/25"
        />
        <SheetHeader className="gap-1 border-b p-4">
          {centred ? (
            <div className="relative">
              {title}
              <SheetCloseButton className="absolute top-1/2 right-0 -translate-y-1/2" />
            </div>
          ) : (
            title
          )}
        </SheetHeader>
        <div className="min-h-0 flex-1" />
      </SheetContent>
    </Sheet>
  );
}

function Harness() {
  const sheet = which === "preview" ? <PreviewSheet /> : <DetailsSheet />;
  return (
    <div
      className="relative h-dvh min-h-0 overflow-hidden bg-background"
      style={CUSTOM_CHROME_STYLE}
    >
      <DesktopChromeVarsEffect
        usesCustomTitlebar={shouldUseCustomWindowTitlebar()}
        usesNativeMacTitlebar={shouldUseNativeMacWindowTitlebar()}
      />
      <WindowTitlebar showSidebarSurface={false} />
      <div className="h-full min-h-0 overflow-hidden">{sheet}</div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Harness />
  </StrictMode>,
);
