// background.js

// 监听网页加载完成事件
chrome.webNavigation.onCommitted.addListener(async (details) => {
  // 【修复核心】只处理主框架 (frameId 为 0)，忽略所有 iframe 和广告窗口
  if (details.frameId !== 0) return;

  const url = details.url;

  if (url.startsWith('http')) {
    console.log(`[Background] 页面加载完成: ${url}，开始检测...`);

    try {
      // ======================
      // 确保 content.js 就绪
      // ======================
      // await chrome.scripting.executeScript({
      //   target: { tabId: details.tabId },
      //   files: ['content/content.js']
      // });
      // await new Promise(resolve => setTimeout(resolve, 100));

      // ======================
      // 1. 处理信号A (串行处理)
      // ======================
      console.log(`[Background] 开始处理信号A...`);
      try {
        const responseA = await fetch('http://localhost:5000/api/scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url }),
        });

        if (!responseA.ok) {
          let errorMsg = `HTTP错误: ${responseA.status}`;
          try { errorMsg = await responseA.text(); } catch (e) {}
          throw new Error(errorMsg);
        }

        const dataA = await responseA.json();
        console.log(`[Background] 信号A检测完成。`);
        chrome.tabs.sendMessage(details.tabId, {
          action: 'show_result',
          model: 'A',
          result: dataA
        });
      } catch (error) {
        console.error(`[Background] 信号A检测失败:`, error);
        chrome.tabs.sendMessage(details.tabId, {
          action: 'show_error',
          error: `信号A检测失败: ${error.message}`
        });
      }

      // ======================
      // 2. 处理信号BC (在A之后串行处理)
      // ======================
      console.log(`[Background] 开始处理信号BC...`);
      // 发送“正在分析BC”的提示
      chrome.tabs.sendMessage(details.tabId, { action: 'show_bc_loading' });

      try {
        const responseBC = await fetch('http://localhost:5001/api/signal_BC', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url }),
        });

        if (!responseBC.ok) {
          let errorMsg = `HTTP错误: ${responseBC.status}`;
          try { errorMsg = await responseBC.text(); } catch (e) {}
          throw new Error(errorMsg);
        }

        const dataBC = await responseBC.json();
        console.log(`[Background] 信号BC检测完成。`);
        chrome.tabs.sendMessage(details.tabId, {
          action: 'show_result',
          model: 'BC',
          result: dataBC
        });
      } catch (error) {
        console.error(`[Background] 信号BC检测失败:`, error);
        chrome.tabs.sendMessage(details.tabId, {
          action: 'show_error',
          error: `信号BC检测失败: ${error.message}`
        });
      }

    } catch (error) {
      // 这个 catch 块处理的是 content.js 注入等最顶层的错误
      console.error(`[Background] 检测流程发生严重错误:`, error);
      chrome.tabs.sendMessage(details.tabId, {
        action: 'show_error',
        error: `检测系统异常: ${error.message}`
      });
    }
  }
}, { url: [{ schemes: ['http', 'https'] }] });