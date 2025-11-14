#!/bin/bash

# Docker credential helper 문제 수정 스크립트
# Usage: ./script/fix-docker-credentials.sh

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Docker Credential Helper 수정${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

CONFIG_FILE="$HOME/.docker/config.json"

# 백업 생성
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}기존 config 백업 중...${NC}"
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${GREEN}✓${NC} 백업 완료: $CONFIG_FILE.backup.*"
    echo ""
fi

# 현재 설정 확인
echo -e "${BLUE}현재 Docker config:${NC}"
cat "$CONFIG_FILE" 2>/dev/null || echo "{}"
echo ""

# credsStore 제거
echo -e "${YELLOW}credsStore 설정 제거 중...${NC}"
if [ -f "$CONFIG_FILE" ]; then
    # jq가 있으면 사용, 없으면 sed 사용
    if command -v jq &> /dev/null; then
        jq 'del(.credsStore)' "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    else
        # credsStore 라인 제거 (간단한 방법)
        sed -i.bak '/"credsStore":/d' "$CONFIG_FILE"
        rm -f "$CONFIG_FILE.bak"
    fi
fi

echo -e "${GREEN}✓${NC} credsStore 설정 제거 완료"
echo ""

# 수정된 설정 확인
echo -e "${BLUE}수정된 Docker config:${NC}"
cat "$CONFIG_FILE"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ 수정 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "이제 다시 시도하세요:"
echo "  ./script/test-docker.sh"
echo ""
echo "문제가 계속되면 Docker Desktop을 재시작하세요:"
echo "  1. Docker Desktop 완전 종료"
echo "  2. Docker Desktop 재시작"
echo "  3. 다시 빌드 시도"
