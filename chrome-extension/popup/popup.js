document.addEventListener('DOMContentLoaded', () => {
  const urlInput = document.getElementById('urlInput');
  const scanButton = document.getElementById('scanButton');
  const resultArea = document.getElementById('resultArea');

  // 页面加载时，自动获取当前标签页的URL
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    if (tabs[0] && tabs[0].url) {
      urlInput.value = tabs[0].url;
    }
  });

  scanButton.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) {
      resultArea.innerHTML = '<p style="color: red;">请输入一个有效的URL。</p>';
      return;
    }

    // 显示加载状态
    scanButton.disabled = true;
    scanButton.textContent = '正在检测...';
    resultArea.innerHTML = '<p>正在分析URL特征并运行AI检测模型...</p>';

    try {
      // 向本地 Flask 服务器发送请求
      const response = await fetch('http://localhost:5000/api/scan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
      });

      const data = await response.json();

      if (response.ok) {
        // 构建结果HTML
        let resultHtml = `<p><strong>检测结果:</strong></p>`;
        resultHtml += `<p>URL: <code>${data.url}</code></p>`;
        resultHtml += `<p>风险等级: <span class="risk-${data.is_risky ? 'high' : 'low'}">${data.risk_label}</span></p>`;
        resultHtml += `<p>风险分数: ${data.risk_score} / 100</p>`;
        resultHtml += `<p>风险概率: ${(data.risk_probability * 100).toFixed(2)}%</p>`;
        
        if (data.reasons && data.reasons.length > 0) {
          resultHtml += `<p><strong>风险原因:</strong></p><ul>`;
          data.reasons.forEach(reason => {
            resultHtml += `<li>${reason}</li>`;
          });
          resultHtml += `</ul>`;
        }

        resultArea.innerHTML = resultHtml;

      } else {
        // 服务器返回错误
        resultArea.innerHTML = `<p style="color: red;">检测失败: ${data.error || '未知错误'}</p>`;
      }

    } catch (error) {
      // 网络错误
      console.error('请求服务器出错:', error);
      resultArea.innerHTML = `<p style="color: red;">无法连接到检测服务。</p><p>请确保 <code>server.py</code> 已在本地启动。</p>`;
    } finally {
      // 恢复按钮状态
      scanButton.disabled = false;
      scanButton.textContent = '开始检测';
    }
  });
});