import os
import logging
from datetime import datetime
import shutil
from selenium.common.exceptions import WebDriverException
from scripts.phishintention.configs import load_config
from scripts.pipeline.test_llm import *
from scripts.utils.PhishIntentionWrapper import LogoDetector, LogoEncoder, LayoutDetector
import yaml
import openai

# --- 全局变量 ---
llm_cls = None
driver = None
param_dict = None
initialized = False

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 配置日志
logging.getLogger("httpcore").setLevel(logging.WARNING)

def init_detector():
    """
    初始化检测器，加载所有必要的模型和启动Selenium Driver。
    这个函数应该在服务器启动时只被调用一次。
    """
    global llm_cls, driver, param_dict, initialized

    if initialized:
        print("检测器已经初始化过了。")
        return True

    print("--- 正在初始化 PhishLLM 检测器 ---")

    # 1. 加载 OpenAI API 密钥
    # 1. 加载 API 密钥
    try:
        # 1. 设置 API Key
        openai_key_path = os.path.join(current_dir, 'datasets', 'openai_key.txt')
        if os.path.exists(openai_key_path):
             # 读取 Key 并直接写入环境变量
             os.environ['OPENAI_API_KEY'] = open(openai_key_path).read().strip()
        else:
             print("警告: 未找到 openai_key.txt")

        # 2. 【核心修改】直接修改环境变量 OPENAI_BASE_URL
        # 这比 openai.base_url = ... 更管用，能强制所有 OpenAI 客户端转向 DeepSeek
        os.environ['OPENAI_BASE_URL'] = "https://api.deepseek.com"
        
        # 3. 双重保险：有些旧库或 LangChain 可能认这个变量
        os.environ['OPENAI_API_BASE'] = "https://api.deepseek.com"
        
        # 4. 清理代理 (防止被梯子带偏)
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        
        print(f"API配置已重定向至: {os.environ['OPENAI_BASE_URL']}")

    except Exception as e:
        print(f"API 配置出错: {e}")
        raise
    # 2. 加载模型和配置
    try:
        default_config = os.path.join(current_dir, 'param_dict.yaml')
        with open(default_config) as file:
            param_dict = yaml.load(file, Loader=yaml.FullLoader)

        AWL_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE = load_config()
        logo_extractor = LogoDetector(AWL_MODEL)
        logo_encoder = LogoEncoder(SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE)
        layout_extractor = LayoutDetector(AWL_MODEL)

        PhishLLMLogger.set_debug_on()
        PhishLLMLogger.set_verbose(False) # 在API模式下，我们通常不希望太多内部日志
        llm_cls = TestVLM(param_dict=param_dict,
                          logo_encoder=logo_encoder,
                          logo_extractor=logo_extractor,
                          layout_extractor=layout_extractor)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

    # 3. 启动 Selenium Driver
    try:
        driver = boot_driver()
        print("Selenium WebDriver 启动成功！")
    except Exception as e:
        print(f"Selenium WebDriver 启动失败: {e}")
        raise

    initialized = True
    print("--- 检测器初始化完成 ---")
    return True

def detect_url(url):
    """
    对单个URL进行检测，并返回结果字典。
    这是提供给外部调用的主要函数。
    """
    global llm_cls, driver

    if not initialized:
        raise Exception("检测器尚未初始化，请先调用 init_detector()。")

    print(f"\n正在处理URL: {url}")

    output_folder = os.path.join(current_dir, "api_results")
    os.makedirs(output_folder, exist_ok=True)
    temp_dir_name = "url_" + datetime.now().strftime("%Y%m%d%H%M%S%f")
    url_folder = os.path.join(output_folder, temp_dir_name)
    os.makedirs(url_folder, exist_ok=True)

    info_path = os.path.join(url_folder, 'info.txt')
    shot_path = os.path.join(url_folder, 'shot.png')
    html_path = os.path.join(url_folder, 'html.txt')

    try:
        with open(info_path, "w", encoding='utf-8') as f:
            f.write(url)

        driver.get(url)
        driver.save_screenshot(shot_path)
        with open(html_path, "w", encoding='utf-8') as f:
            f.write(driver.page_source)

        logo_box, reference_logo = llm_cls.detect_logo(shot_path)
        pred, F_brand, F_intent, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis = llm_cls.test(
            url=url,
            reference_logo=reference_logo,
            logo_box=logo_box,
            shot_path=shot_path,
            html_path=html_path,
            driver=driver,
        )
        driver.delete_all_cookies()

        result = {
            "url": url,
            "prediction": pred,
            "is_phish": pred == 'phish',
            "F_brand_flag": F_brand,
            "F_intent_flag": F_intent,
            "brand_recognition_time": round(brand_recog_time, 4),
            "crp_prediction_time": round(crp_prediction_time, 4),
            "crp_transition_time": round(crp_transition_time, 4),
        }
        return result

    except Exception as e:
        print(f"检测URL {url} 时发生错误: {e}")
        raise  # 将错误向上抛出，让调用者处理

def cleanup_detector():
    """
    清理检测器资源，特别是关闭Selenium Driver。
    这个函数应该在服务器关闭时被调用。
    """
    global driver, initialized
    if driver:
        try:
            driver.quit()
            print("Selenium WebDriver 已成功关闭。")
        except Exception as e:
            print(f"关闭Selenium WebDriver时发生错误: {e}")
    initialized = False

def restart_driver():
    """
    重启Selenium Driver，用于在Driver崩溃后恢复服务。
    """
    global driver
    print("尝试重启 Selenium Driver...")
    try:
        if driver:
            driver.quit()
    except:
        pass
    driver = boot_driver()
    print("Selenium Driver 重启成功。")
    return driver