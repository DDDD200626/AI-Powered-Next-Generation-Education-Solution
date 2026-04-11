# Linux/macOS/WSL: make stack
.PHONY: stack stack-down stack-logs stack-check
stack:
	docker compose up --build -d
	@echo ""
	@echo "Site: http://127.0.0.1:8080"
	@echo "API:  http://127.0.0.1:8000/docs"

stack-down:
	docker compose down

stack-logs:
	docker compose logs -f web api

stack-check:
	@chmod +x scripts/check-stack.sh 2>/dev/null; ./scripts/check-stack.sh
