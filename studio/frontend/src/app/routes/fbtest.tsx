// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Throwaway staging-only route: renders the FolderBrowser dialog directly so a
// Playwright driver can exercise the folder-browser + drive-hopping UI without
// clicking through the multi-step model picker. NOT for upstream.
import { FolderBrowser } from "@/components/assistant-ui/model-selector/folder-browser";
import { createRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Route as rootRoute } from "./__root";

function FbTestPage() {
  const [open, setOpen] = useState(true);
  const [picked, setPicked] = useState<string>("");
  return (
    <div className="p-6" data-testid="fbtest-root">
      <button
        type="button"
        data-testid="fbtest-open"
        onClick={() => setOpen(true)}
      >
        Open folder browser
      </button>
      <div data-testid="fbtest-picked">{picked}</div>
      <FolderBrowser
        open={open}
        onOpenChange={setOpen}
        onSelect={(p) => setPicked(p)}
      />
    </div>
  );
}

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/fbtest",
  component: FbTestPage,
});
