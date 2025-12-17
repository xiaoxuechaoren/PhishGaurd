// content.js - 调试修复版

// 全局变量
var bcLoadingNotification = null;

console.log("[Content] 脚本已加载，准备就绪。");

// 监听消息
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("[Content] 收到消息:", request);

  try {
    if (request.action === 'show_result') {
      const result = request.result;
      const model = request.model;

      if (model === 'A') {
        displayResultA(result);
      } else if (model === 'BC') {
        console.log("[Content] 准备显示 BC 结果...");
        displayResultBC(result);
      }
    } else if (request.action === 'show_error') {
      displayError(request.error);
    } else if (request.action === 'show_bc_loading') {
      showBCLoadingNotification();
    }
  } catch (e) {
    console.error("[Content] 处理消息时发生错误:", e);
  }
});

// 显示模型 A 的结果
function displayResultA(result) {
  // 先移除旧的
  const old = document.getElementById('result-a-notification');
  if (old) old.remove();

  const notification = document.createElement('div');
  notification.id = 'result-a-notification'; 

  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    z-index: 2147483647; /* 最大的 z-index */
    font-family: sans-serif;
    max-width: 350px;
    background-color: #f5fafe;
    border-left: 5px solid #3b82f6;
    color: #333;
  `;

  notification.innerHTML = `
    <div style="font-weight: bold; margin-bottom: 5px; color: #1d4ed8;">⚡ 快速检测结果</div>
    <div style="font-size: 13px;">风险等级: ${result.risk_label}</div>
    <div style="font-size: 13px;">分数: ${result.risk_score}</div>
  `;

  document.body.appendChild(notification);
}

// 显示 BC 加载中
function showBCLoadingNotification() {
  // 1. 暴力清理旧的
  const old = document.getElementById('bc-loading-popup');
  if (old) old.remove();

  console.log("[Content] 创建加载弹窗...");

  bcLoadingNotification = document.createElement('div');
  bcLoadingNotification.id = 'bc-loading-popup';
  
  // 暂时固定在 top: 120px，防止位置计算错误导致不显示
  bcLoadingNotification.style.cssText = `
    position: fixed;
    top: 120px; 
    right: 20px;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    z-index: 2147483647 !important;
    font-family: sans-serif;
    max-width: 350px;
    background-color: #fffbeb;
    border-left: 5px solid #f59e0b;
    color: #333;
    transition: all 0.3s;
  `;

  bcLoadingNotification.innerHTML = `
    <div style="font-weight: bold; margin-bottom: 5px; color: #b45309;">🔍 正在深度分析...</div>
    <div style="font-size: 12px;">请稍候，正在进行视觉与内容核验</div>
  `;

  document.body.appendChild(bcLoadingNotification);
}

// 显示 BC 最终结果
function displayResultBC(result) {
  console.log("[Content] 进入 displayResultBC 函数");

  // 1. 再次暴力清理加载弹窗 (确保它一定消失)
  const loadingPopup = document.getElementById('bc-loading-popup');
  if (loadingPopup) {
    console.log("[Content] 移除加载弹窗成功");
    loadingPopup.remove();
  } else {
    console.log("[Content] 未发现加载弹窗，跳过移除");
  }
  bcLoadingNotification = null;

  // 2. 创建新弹窗
  const notification = document.createElement('div');
  notification.id = 'bc-result-popup'; // 给它个 ID 方便调试

  const isPhish = result.is_phish;
  const bgColor = isPhish ? '#fff5f5' : '#f0fdf4';
  const borderColor = isPhish ? '#e53e3e' : '#10b981';
  const titleText = isPhish ? '⚠️ 高风险 URL' : '✅ 安全 URL';
  const titleColor = isPhish ? '#c53030' : '#047857';

  // 3. 简化样式，强制位置，防止计算错误跑飞
  notification.style.cssText = `
    position: fixed;
    top: 120px; /* 固定位置，不依赖计算 */
    right: 20px;
    padding: 15px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    z-index: 2147483647;
    font-family: sans-serif;
    max-width: 350px;
    background-color: ${bgColor};
    border-left: 5px solid ${borderColor};
    color: #333;
    animation: fadeIn 0.5s;
  `;

  notification.innerHTML = `
    <div style="font-weight: bold; margin-bottom: 8px; font-size: 15px; color: ${titleColor};">
      ${titleText}
    </div>
    <div style="font-size: 13px; margin-bottom: 4px;">检测结果: <strong>${result.prediction}</strong></div>
    <div style="font-size: 12px; opacity: 0.8;">品牌异常: ${result.F_brand_flag ? '是' : '否'}</div>
    <div style="font-size: 12px; opacity: 0.8;">意图可疑: ${result.F_intent_flag ? '是' : '否'}</div>
    <button id="bc-close-btn" style="margin-top: 10px; padding: 5px 10px; border:none; background:rgba(0,0,0,0.05); cursor:pointer; border-radius:4px;">关闭</button>
  `;

  // 4. 插入 DOM
  document.body.appendChild(notification);
  console.log("[Content] 结果弹窗已插入 DOM");

  // 5. 绑定关闭事件
  setTimeout(() => {
    const btn = document.getElementById('bc-close-btn');
    if (btn) {
      btn.onclick = function() {
        notification.remove();
      };
    }
  }, 100);
}

function displayError(msg) {
  alert("PhishGuard 错误: " + msg);
}