# Frontend Production Dockerfile
# Multi-stage build for optimized production bundle

FROM node:20-alpine as builder

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production=false

# Copy source code
COPY frontend/ .

# Build production bundle
RUN npm run build

# Production stage with Nginx
FROM nginx:alpine

# Copy custom Nginx configuration
COPY infrastructure/nginx/nginx.conf /etc/nginx/nginx.conf
COPY infrastructure/nginx/default.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Create non-root user
RUN addgroup -g 1000 vokg && \
    adduser -D -u 1000 -G vokg vokg && \
    chown -R vokg:vokg /usr/share/nginx/html && \
    chown -R vokg:vokg /var/cache/nginx && \
    chown -R vokg:vokg /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R vokg:vokg /var/run/nginx.pid

USER vokg

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:80/health || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
