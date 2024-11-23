import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/offer': {
        target: 'http://localhost:8080',
        changeOrigin: true
      },
      '/update-config': {
        target: 'http://localhost:8080',
        changeOrigin: true
      },
      '/metrics': {
        target: 'http://localhost:8080',
        changeOrigin: true
      },
      '/supported-resolutions': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  }
});