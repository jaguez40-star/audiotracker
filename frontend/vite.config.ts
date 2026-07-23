import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 6023,
    // El proxy evita CORS en desarrollo y deja las llamadas del cliente como
    // rutas relativas, sin URLs absolutas repartidas por el codigo.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:6024',
        changeOrigin: true,
      },
    },
  },
})
