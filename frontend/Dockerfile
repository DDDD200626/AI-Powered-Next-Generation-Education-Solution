# 정적 빌드 + nginx (docker-compose에서 /api 는 API 컨테이너로 프록시)
# VITE_API_BASE 는 빈 값이면 브라우저가 같은 출처 /api 로 호출 → 컨테이너 nginx 가 api 로 프록시
FROM node:20-alpine AS build
WORKDIR /app
ARG VITE_API_BASE=
ENV VITE_API_BASE=$VITE_API_BASE
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
