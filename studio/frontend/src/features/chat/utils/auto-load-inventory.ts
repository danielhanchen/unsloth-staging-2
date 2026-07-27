// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The two rules the send-time auto-load sweep needs beyond its own loops (#7374):
 * which on-device row it may pick unattended, and what to say when it picked
 * nothing.
 */

/** The `GET /api/hub/local` row fields the sweep reads. */
export type AutoLoadInventoryRow = {
  runtime?: string | null;
  partial?: boolean | null;
  capabilities?: { can_chat?: boolean | null } | null;
};

/**
 * True when the scanner already vouched for this row as a loadable chat model.
 *
 * Auto-load picks with no confirmation, so it trusts the capability the backend
 * computed while it stat-ed the directory rather than guessing from filenames or
 * folder shape: `can_chat` is already false for partials, projector-only folders
 * and anything unclassified, and the runtime check leaves LoRA adapters (not a
 * standalone model) an explicit user pick.
 */
export function isAutoLoadableLocalRow(row: AutoLoadInventoryRow): boolean {
  if (!row?.capabilities?.can_chat || row.partial) return false;
  const runtime = (row.runtime ?? "").trim().toLowerCase();
  return runtime === "llama_cpp" || runtime === "transformers";
}

export type AutoLoadFailure = { title: string; description: string };

/**
 * What to tell the user when the sweep loaded nothing. "No downloaded models
 * found" used to be printed for every outcome, including ones where models were
 * found and then skipped, which is what #7374 read as Studio not seeing a model
 * that was on disk. Report the state that actually held.
 */
export function describeAutoLoadFailure(input: {
  candidateCount: number;
  inventoryUnavailable: boolean;
  blockedByTrustRemoteCode: boolean;
  lastFailureReason?: string | null;
}): AutoLoadFailure {
  if (input.blockedByTrustRemoteCode) {
    return {
      title: "This model needs custom code approval",
      description:
        "Select it from the top bar to review and approve its custom code, or pick another model.",
    };
  }
  if (input.inventoryUnavailable) {
    return {
      title: "Could not read the on-device model list",
      description:
        "Studio could not check which models are downloaded, so it did not pick one. Retry, or select a model in the top bar.",
    };
  }
  if (input.candidateCount > 0) {
    const reason = input.lastFailureReason?.trim();
    return {
      title:
        input.candidateCount === 1
          ? "The downloaded model could not be loaded"
          : `None of the ${input.candidateCount} downloaded models could be loaded`,
      description: reason
        ? `${reason.endsWith(".") ? reason : `${reason}.`} Select a model in the top bar to load it yourself.`
        : "Select a model in the top bar to load it yourself.",
    };
  }
  return {
    title: "No models on this device",
    description:
      "Download a model from the Hub, or add the folder that holds your models under Settings, then retry.",
  };
}
