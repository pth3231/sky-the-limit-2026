"use client";

import { DraftWorkspace } from "../components/DraftWorkspace/DraftWorkspace";
import { useDraftSession } from "../lib/session";

/** The coach's single-screen series and draft workspace. */
export default function Home() {
  const session = useDraftSession();
  return <DraftWorkspace {...session} />;
}