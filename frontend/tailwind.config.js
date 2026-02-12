/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        midnight: '#0a0e1a',
        abyss: '#0d1224',
        slate: {
          850: '#141c2e',
          750: '#1e293b',
        },
        nebula: {
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
        },
        aurora: {
          400: '#34d399',
          500: '#10b981',
        },
        ember: {
          400: '#fb923c',
          500: '#f97316',
        },
        cyan: {
          400: '#22d3ee',
          500: '#06b6d4',
        },
      },
      fontFamily: {
        display: ['Instrument Serif', 'Georgia', 'serif'],
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
};
