#!/bin/bash

BASE_URL="http://ivi-cn010:8001"

response=$(curl -s $BASE_URL/retriever_info)

echo "Retriever info:"
echo "$response"
