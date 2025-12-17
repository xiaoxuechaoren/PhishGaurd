#解决报错
from flask import Flask, request, jsonify
from flask_cors import CORS
import signal_A  

app = Flask(__name__)
CORS(app)
# --- 关键步骤 ---
# 在服务器启动时，预先加载模型、Tokenizer和Scaler
# 这样它们只会被加载一次，保存在内存中，所有请求都可以复用
print("正在加载检测模型和相关组件...")
try:
    model, tokenizer, scaler = signal_A.load_detection_model()
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    # 如果加载失败，让服务器启动失败
    raise e

# --- API 端点定义 ---
@app.route('/api/scan', methods=['POST'])
def scan_url():
    """
    接收来自 Chrome 扩展的 URL 扫描请求
    """
    try:
        # 1. 从请求中获取 JSON 数据
        data = request.get_json()
        
        # 2. 检查 URL 是否存在
        if not data or 'url' not in data:
            return jsonify({"error": "请求体中缺少 'url' 字段"}), 400
        
        url = data['url']
        print(f"收到扫描请求: {url}")

        # 3. 调用预测函数进行检测
        # 注意：这里直接使用了预先加载好的 model, tokenizer, scaler
        # ...
        label, prob = signal_A.predict_phishing(url, model, tokenizer, scaler)

        prob = float(prob)

        response = {
            "url": url,
            "is_risky": label == "钓鱼URL",
            "risk_label": label,
            "risk_score": round(prob * 100, 2),
            "risk_probability": round(prob, 4),
            "reasons": [
                f"风险概率超过 50%，判定为{label}",
                "检测基于机器学习模型和URL特征分析"
            ]
        }

        print(f"扫描结果: {response}")
        
        return jsonify(response)

    except Exception as e:
        # 错误处理
        print(f"扫描过程中发生错误: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 启动服务器，确保 host='0.0.0.0' 以便外部访问
    app.run(host='0.0.0.0', port=5000, debug=True)