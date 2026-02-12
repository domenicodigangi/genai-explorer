import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'GenAI Knowledge Explorer',
  description: 'Navigate the generative AI landscape through conversational RAG and interactive topic visualization',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
