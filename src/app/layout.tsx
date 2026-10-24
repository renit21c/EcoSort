import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'Ecosort', description: 'AI-powered waste classification and recycling guidance.' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
