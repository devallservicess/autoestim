# ── AutoEstim v3 — Frontend Dockerfile ─────────────────────────────────────
# Nginx ultra-léger pour servir les fichiers statiques HTML/CSS/JS
FROM nginx:alpine

# Copier les fichiers frontend dans nginx
COPY . /usr/share/nginx/html

# Configuration nginx personnalisée (gzip, cache, CORS)
RUN echo 'server { \
    listen 80; \
    root /usr/share/nginx/html; \
    index index.html; \
    gzip on; \
    gzip_types text/css application/javascript; \
    location / { try_files $uri $uri/ /index.html; } \
    location ~* \.(js|css|html)$ { expires 1h; add_header Cache-Control "public"; } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80
