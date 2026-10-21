import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
// The /api proxy only applies to `npm run dev` / `npm run preview` (local development);
// in production the built files are served behind the existing /api reverse proxy.
const api = { '/api': { target: process.env.DASHBOARD_API || 'http://127.0.0.1:8000', changeOrigin: true } }

export default defineConfig({
  plugins: [react()],
  server: { proxy: api },
  preview: { proxy: api },
})
