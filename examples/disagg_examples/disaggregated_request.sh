# serve two example requests
output1=$(curl -X POST -s http://localhost:8006/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/share/models/Llama-3-8B-Instruct-Gradient-1048k",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0,
"pd_pair": [0,3]
}')

echo ""
echo "Output of first request: $output1"

echo "🎉🎉 Successfully finished 1 test requests! 🎉🎉"
echo ""