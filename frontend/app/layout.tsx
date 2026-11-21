import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Draft Coach",
  description: "A draft preparation workspace for Cloud9 coaches.",
};

/** Root document for the draft workspace. */
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}