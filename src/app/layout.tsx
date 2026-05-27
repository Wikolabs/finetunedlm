import type { Metadata } from "next";
import { Space_Grotesk, Inter } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-display", display: "swap" });
const inter = Inter({ subsets: ["latin"], variable: "--font-body", display: "swap" });

export const metadata: Metadata = {
  title: "FineTuneDLM -- Votre LLM maison fine-tune en 48h",
  description: "Fine-tuning LoRA/QLoRA sur vos donnees -- LLM prive et souverain deploye sur votre infrastructure en 48h, 10x moins cher que le cloud.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" className={${spaceGrotesk.variable} ${inter.variable}}>
      <body style={{ fontFamily: "var(--font-body)", background: "#fff7ed" }}>{children}</body>
    </html>
  );
}