// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* eslint-disable no-restricted-imports -- Exercise the real hook and UI without the feature barrel. */

import { Toaster } from "@/components/ui/sonner";
import { ModelLoadInlineStatus } from "@/features/chat/components/model-load-status";
import { useChatModelRuntime } from "@/features/chat/hooks/use-chat-model-runtime";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { createRoot } from "react-dom/client";
import "./styles.css";

// Review-only: Selenium cannot intercept the network the way Playwright routes do, so the same
// canned responses the Playwright fixture serves are installed in-page instead. The hook, the
// toast, the component and the CSS under test are all still the real ones.
const realFetch = window.fetch.bind(window);
window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const path = new URL(url, location.origin).pathname;
  if (!path.startsWith("/api/")) return realFetch(input as never, init);
  const json = (body: unknown) =>
    new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  if (path === "/api/inference/load") return new Promise<Response>(() => {}); // never settles
  if (path === "/api/inference/load-progress")
    return json({ phase: "mmap", bytes_total: 1_000_000_000, bytes_loaded: 420_000_000, fraction: 0.42 });
  if (path === "/api/models/download-progress")
    return json({ expected_bytes: 1_000_000_000, downloaded_bytes: 420_000_000, progress: 0.42 });
  if (path === "/api/inference/status")
    return json({ active_model: null, loaded: [], loading: [], is_gguf: true, gguf_variant: "Q4_K_M" });
  if (path === "/api/inference/unload") return json({ status: "unloaded" });
  if (path === "/api/models/list") return json({ models: [] });
  if (path === "/api/models/loras") return json({ loras: [] });
  if (path === "/api/inference/validate") return json({ valid: true, is_gguf: true });
  if (path === "/api/inference/active-generations") return json({ active_generations: [] });
  if (path.includes("gpu")) return json({ gpus: [] });
  return json({});
};

localStorage.setItem("unsloth_auth_token", "model-load-fixture");
useChatRuntimeStore.setState({ settingsHydrated: true });
// eslint-disable-next-line react-refresh/only-export-components -- Standalone browser fixture.
function App() {
  const runtime = useChatModelRuntime();
  const loading = useChatRuntimeStore((s) => s.modelLoading);
  return (
    <main className="p-8">
      <button
        type="button"
        onClick={() =>
          runtime.selectModel({
            id: "fixture/model",
            isGguf: true,
            isDownloaded: !new URLSearchParams(location.search).has("download"),
            ggufVariant: "Q4_K_M",
            forceReload: true,
          })
        }
      >
        Load model
      </button>
      <p data-testid="lifecycle">{loading ? "Busy" : "Idle"}</p>
      <div data-testid="inline-status">
        {runtime.loadingModel && runtime.loadToastDismissed ? (
          <ModelLoadInlineStatus
            label="Loading model…"
            title="Loading model"
            progressPercent={runtime.loadProgress?.percent}
            progressLabel={runtime.loadProgress?.label}
            onStop={runtime.cancelLoading}
          />
        ) : null}
      </div>
      <Toaster />
    </main>
  );
}
const root = document.getElementById("root");
if (!root) {
  throw new Error("Missing fixture root");
}
createRoot(root).render(<App />);
