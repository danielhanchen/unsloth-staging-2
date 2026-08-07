// Throwaway harness for unslothai/unsloth#7954 on top of the merged #7957.
// Staging only, never committed to the PR. Drives the REAL SheetContent with
// each of the seven call sites' exact props, so the geometry measured here is
// the geometry the app produces.
//
// ?site=1..7        the call site (see SITES below)
// ?variant=naive    both offset mechanisms live (the bad conflict resolution)
// ?variant=fixed    only #7954's shared class (per-sheet compensation removed)
// ?chrome=custom    Windows/Linux desktop: 34px inset + real WindowTitlebar
// ?chrome=mac       macOS desktop: native titlebar, var unset so 0px applies
// ?chrome=browser   browser mode: no chrome at all, var unset so 0px applies
// ?fs=12..20        UI font size
import { StrictMode, useEffect, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import { WindowTitlebar } from "@/components/tauri/window-titlebar";
import {
  Sheet,
  SheetCloseButton,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

const params = new URLSearchParams(window.location.search);
const site = Number(params.get("site") ?? "1");
const variant = params.get("variant") ?? "fixed";
const chrome = params.get("chrome") ?? "custom";
const fontSize = Number(params.get("fs") ?? "15");

if (fontSize !== 15) {
  document.documentElement.style.setProperty(
    "--ui-font-scale",
    String(fontSize / 16),
  );
  document.documentElement.setAttribute("data-ui-font-size", String(fontSize));
}
document.documentElement.style.removeProperty("font-size");

// Verbatim from app/provider.tsx.
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

const MAC_CHROME_STYLE = {
  "--studio-titlebar-height": "0px",
  "--studio-mac-titlebar-height": "34px",
  "--studio-content-top-inset": "34px",
} as CSSProperties;

// Verbatim behaviour of app/provider.tsx's DesktopChromeVarsEffect. On the
// custom-titlebar path it is always mounted: the sheet class now reads the
// mirrored var off <html>, so this IS the mechanism under test.
function DesktopChromeVarsEffect({ active }: { active: boolean }) {
  useEffect(() => {
    if (!active) return;
    const el = document.documentElement;
    el.style.setProperty("--studio-custom-titlebar-height", "34px");
    el.style.setProperty("--studio-window-control-inset", "112px");
    return () => {
      el.style.removeProperty("--studio-custom-titlebar-height");
      el.style.removeProperty("--studio-window-control-inset");
    };
  }, [active]);
  return null;
}

// The compensation #7957 added to sites 1 and 5, live only in `naive`.
const legacyOffset: CSSProperties =
  variant === "naive"
    ? {
        height: "calc(100% - var(--studio-custom-titlebar-height, 0px))",
        marginTop: "var(--studio-custom-titlebar-height, 0px)",
      }
    : {};

// Only sites 1, 3 and 5 suppress the default close button and place their own;
// the rest keep SheetContent's, so adding a second one here would have them
// occlude each other and report a false positive.
function Body({ label, ownClose }: { label: string; ownClose?: boolean }) {
  const title = (
    <SheetTitle className="flex items-center gap-2 pr-10 font-heading text-base">
      {label}
    </SheetTitle>
  );
  return (
    <>
      <SheetHeader className="border-b p-4">
        {ownClose ? (
          <div className="relative">
            {title}
            <SheetCloseButton className="absolute top-1/2 right-0 -translate-y-1/2" />
          </div>
        ) : (
          title
        )}
      </SheetHeader>
      <div className="min-h-0 flex-1" />
    </>
  );
}

type Site = {
  label: string;
  render: (container: HTMLDivElement | null) => ReactNode;
};

const SITES: Record<number, Site> = {
  1: {
    label: "response details",
    render: () => (
      <SheetContent
        side="right"
        className="w-[min(28rem,100vw)] p-0 sm:max-w-[28rem]"
        showCloseButton={false}
        style={legacyOffset}
      >
        <Body label="Response details" ownClose={true} />
      </SheetContent>
    ),
  },
  2: {
    label: "chat settings (mobile)",
    render: () => (
      <SheetContent side="right" className="w-[18rem] p-0 font-heading">
        <Body label="Chat settings" />
      </SheetContent>
    ),
  },
  3: {
    label: "deep research",
    render: () => (
      <SheetContent
        side="right"
        className="w-screen max-w-none p-0 sm:max-w-none"
        showCloseButton={false}
      >
        {/* research-activity-panel's variant="sheet" aside, post-#7954 (no offset) */}
        <aside className="relative flex h-full min-h-0 flex-col bg-background text-foreground">
          <Body label="Deep research" ownClose={true} />
        </aside>
      </SheetContent>
    ),
  },
  4: {
    label: "sidebar (mobile)",
    render: () => (
      <SheetContent
        side="left"
        className="bg-sidebar text-sidebar-foreground w-2/3 max-w-[18rem] p-0 [&>button]:hidden"
      >
        <Body label="Sidebar" />
      </SheetContent>
    ),
  },
  5: {
    label: "document preview",
    render: () => (
      <SheetContent
        side="right"
        style={{ width: 704, maxWidth: "95vw", ...legacyOffset }}
        className="flex w-full flex-col gap-0 p-0"
        showCloseButton={false}
      >
        <Body label="quarterly-report.pdf" ownClose={true} />
      </SheetContent>
    ),
  },
  6: {
    label: "recipe block (in-container, absolute)",
    render: (container) => (
      <SheetContent
        side="right"
        position="absolute"
        overlayPosition="absolute"
        container={container}
        className="absolute gap-0 p-0 shadow-none"
      >
        <Body label="Block" />
      </SheetContent>
    ),
  },
  7: {
    label: "chart settings",
    render: () => (
      <SheetContent className="w-full sm:max-w-md">
        <Body label="Chart settings" />
      </SheetContent>
    ),
  },
};

function Harness() {
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  const entry = SITES[site] ?? SITES[1];

  const sheet = <Sheet open={true}>{entry.render(container)}</Sheet>;

  // Site 6 lives inside a route-local relative container, like recipe-studio.
  // The root route (app/routes/__root.tsx) pads non-chat routes down by the
  // titlebar inset, and the container is a flow child of that padded wrapper,
  // so the padding has to sit on an OUTER div: an absolutely positioned sheet
  // resolves against its containing block's padding box and would ignore
  // padding applied to the container itself.
  const content =
    site === 6 ? (
      <div
        className="h-full min-h-0 overflow-hidden"
        style={{ paddingTop: "var(--studio-content-top-inset, 0px)" }}
      >
        <div
          ref={setContainer}
          className="relative h-full min-h-0 overflow-hidden"
        >
          {container ? sheet : null}
        </div>
      </div>
    ) : (
      sheet
    );

  if (chrome === "browser") {
    return <div className="h-dvh min-h-0 overflow-hidden">{content}</div>;
  }

  if (chrome === "mac") {
    return (
      <div
        className="relative h-dvh min-h-0 overflow-hidden bg-background"
        style={MAC_CHROME_STYLE}
      >
        {content}
      </div>
    );
  }

  return (
    <div
      className="relative h-dvh min-h-0 overflow-hidden bg-background"
      style={CUSTOM_CHROME_STYLE}
    >
      <DesktopChromeVarsEffect active={true} />
      <WindowTitlebar showSidebarSurface={false} />
      <div className="h-full min-h-0 overflow-hidden">{content}</div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Harness />
  </StrictMode>,
);
