# ONZE frontend image: React + Vite dev server (spec §5).
# The React app itself is scaffolded by the frontend-engineer; this image only
# provides a reproducible Node 22 runtime. It expects a package.json to exist.
FROM node:22-slim

WORKDIR /app

# Install deps first for layer caching. `|| true` lets the image build before
# the frontend is scaffolded; real installs happen once package.json lands.
COPY package*.json ./
RUN npm install || true

COPY . .

EXPOSE 5173

# --host exposes Vite outside the container.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
