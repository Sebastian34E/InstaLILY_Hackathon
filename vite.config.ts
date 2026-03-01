import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ws': {
        target: 'wss://5a02-34-133-228-240.ngrok-free.app',
        ws: true,
        changeOrigin: true,
        secure: false,
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      },
      '/problems': {
        target: 'https://5a02-34-133-228-240.ngrok-free.app',
        changeOrigin: true,
        secure: false,
        headers: {
          'ngrok-skip-browser-warning': 'true',
        },
      },
    },
  },
})
