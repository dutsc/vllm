# serve two example requests
output1=$(curl -X POST -s http://localhost:8006/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/share/models/Meta-Llama-3-8B-Instruct",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0,
"pd_pair": [0,1]
}')

echo ""
echo "Output of first request: $output1"

echo "🎉🎉 Successfully finished 1 test requests! 🎉🎉"
echo ""