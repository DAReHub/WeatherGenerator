# ---- Build Stage ----
FROM node:22.12-slim AS build

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

ARG VITE_API_URL
ARG VITE_MAX_REALS
ARG VITE_MAX_LOCATIONS

ENV VITE_API_URL=$VITE_API_URL
ENV VITE_MAX_REALS=$VITE_MAX_REALS
ENV VITE_MAX_LOCATIONS=$VITE_MAX_LOCATIONS

RUN npx vite build

# ---- Production Stage ----
FROM nginx:stable-alpine AS production

COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]