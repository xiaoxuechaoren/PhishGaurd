# server_BC.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import scripts.infer.signal_BC as signal_BC  # 导入你的检测器模块

app = Flask(__name__)
CORS(app)
# --- 关键步骤 ---
# 在服务器启动时，预先加载模型、Tokenizer和Scaler
# 这样它们只会被加载一次，保存在内存中，所有请求都可以复用
print("正在加载检测模型和相关组件...")
try:
    signal_BC.init_detector()
    print("模型BC加载成功！")
except Exception as e:
    print(f"模型BC加载失败: {e}")
    # 如果加载失败，让服务器启动失败
    raise e

# --- API 端点定义 ---
@app.route('/api/signal_BC', methods=['POST'])
def scan_url():
    """
    接收来自客户端的 URL 扫描请求
    """
    try:
        # 1. 从请求中获取 JSON 数据
        data = request.get_json()
        
        # 2. 检查 URL 是否存在
        if not data or 'url' not in data:
            return jsonify({"error": "请求体中缺少 'url' 字段"}), 400
        
        url = data['url']
        print(f"收到扫描请求: {url}")

        # 3. 调用 signal_BC 中的预测函数进行检测
        # 注意：这里直接使用了预先初始化好的检测器
        detection_result = signal_BC.detect_url(url)

        # 打印检测结果
        print(f"检测结果: {detection_result}")

        # 4. 返回成功响应
        return jsonify(detection_result)

    except Exception as e:
        # 错误处理
        print(f"扫描过程中发生错误: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 启动服务器，确保 host='0.0.0.0' 以便外部访问
     app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)