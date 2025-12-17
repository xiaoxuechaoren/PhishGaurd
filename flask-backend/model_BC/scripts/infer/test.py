from datetime import datetime, date, timedelta
from scripts.phishintention.configs import load_config
from scripts.pipeline.test_llm import *
from scripts.utils.PhishIntentionWrapper import LogoDetector, LogoEncoder, LayoutDetector
import argparse
from tqdm import tqdm
import yaml
import openai
import logging
from selenium.common.exceptions import *
import os
import tempfile
import shutil

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 使用绝对路径设置API密钥
openai_key_path = os.path.join(current_dir, 'datasets', 'openai_key.txt')
os.environ['OPENAI_API_KEY'] = open(openai_key_path).read().strip()

logging.getLogger("httpcore").setLevel(logging.WARNING)

def process_single_url(url, llm_cls, driver, output_folder=None):
    """
    处理单个URL，生成必要的文件（info.txt, shot.png），然后调用PhishLLM进行检测。
    """
    print(f"\n正在处理单个URL: {url}")

    # 设置输出文件夹为绝对路径
    if output_folder is None:
        output_folder = os.path.join(current_dir, "single_url_results")
    
    # 创建一个临时目录来存放该URL的相关文件
    os.makedirs(output_folder, exist_ok=True)
    temp_dir_name = "single_url_" + datetime.now().strftime("%Y%m%d%H%M%S")
    url_folder = os.path.join(output_folder, temp_dir_name)
    os.makedirs(url_folder, exist_ok=True)

    info_path = os.path.join(url_folder, 'info.txt')
    shot_path = os.path.join(url_folder, 'shot.png')
    html_path = os.path.join(url_folder, 'html.txt')
    predict_path = os.path.join(url_folder, 'predict.png')

    # 1. 保存URL到info.txt
    with open(info_path, "w", encoding='utf-8') as f:
        f.write(url)

    # 2. 使用Selenium driver获取网页截图和HTML
    try:
        driver.get(url)
        # 等待页面加载，可以根据需要调整等待策略
        # WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # 保存截图
        driver.save_screenshot(shot_path)
        print(f"截图已保存到: {shot_path}")

        # 保存HTML源码
        with open(html_path, "w", encoding='utf-8') as f:
            f.write(driver.page_source)
        print(f"HTML源码已保存到: {html_path}")

    except Exception as e:
        print(f"获取URL {url} 的信息时出错: {e}")
        # 清理临时文件
        shutil.rmtree(url_folder)
        return None

    # 3. 调用PhishLLM进行检测
    try:
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

        if pred == 'phish':
            plotvis.save(predict_path)
            print(f"预测可视化图已保存到: {predict_path}")

        # 4. 组织并返回结果
        result = {
            "url": url,
            "prediction": pred,
            "F_brand_flag": F_brand,
            "F_intent_flag": F_intent,
            "brand_recog_time": brand_recog_time,
            "crp_prediction_time": crp_prediction_time,
            "crp_transition_time": crp_transition_time,
            "result_folder": url_folder
        }

        return result

    except Exception as e:
        print(f"使用PhishLLM检测URL {url} 时出错: {e}")
        # 清理临时文件
        shutil.rmtree(url_folder)
        return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PhishLLM Detector - 支持批量文件夹或单个URL检测")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", help="包含网站数据的文件夹路径 (例如: ./datasets/field_study/2023-09-02/)")
    group.add_argument("--url", help="需要检测的单个URL (例如: https://example.com)")
    
    # 使用绝对路径作为默认配置路径
    default_config = os.path.join(current_dir, 'param_dict.yaml')
    parser.add_argument("--config", default=default_config, help="Config .yaml path")
    args = parser.parse_args()

    PhishLLMLogger.set_debug_on()
    PhishLLMLogger.set_verbose(True)

    # load hyperparameters
    with open(args.config) as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    AWL_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE = load_config()
    logo_extractor = LogoDetector(AWL_MODEL)
    logo_encoder = LogoEncoder(SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE)
    layout_extractor = LayoutDetector(AWL_MODEL)

    # PhishLLM
    llm_cls = TestVLM(param_dict=param_dict,
                      logo_encoder=logo_encoder,
                      logo_extractor=logo_extractor,
                      layout_extractor=layout_extractor)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = os.getenv("http_proxy", "")  # set openai proxy

    driver = boot_driver()

    # --- 统一输入格式的核心逻辑 ---
    if args.url:
        # 处理单个URL
        result = process_single_url(args.url, llm_cls, driver)
        if result:
            print("\n--- 检测结果 ---")
            print(f"URL: {result['url']}")
            print(f"预测结果: {result['prediction']}")
            print(f"品牌意图不符标志 (F_brand): {result['F_brand_flag']}")
            print(f"凭证窃取意图标志 (F_intent): {result['F_intent_flag']}")
            print(f"结果文件保存在: {result['result_folder']}")
        else:
            print(f"\n处理URL {args.url} 失败。")
            
    elif args.folder:
        # 保持原有的批量处理逻辑
        day = date.today().strftime("%Y-%m-%d")
        # 使用绝对路径保存结果文件
        result_txt = os.path.join(current_dir, '{}_phishllm.txt'.format(day))

        if not os.path.exists(result_txt):
            with open(result_txt, "w+", encoding='utf-8') as f:
                f.write("folder" + "\t")
                f.write("url" + "\t") # 新增一列URL
                f.write("phish_prediction" + "\t")
                f.write("F_brand_flag" + "\t")
                f.write("F_intent_flag" + "\t")
                f.write("brand_recog_time" + "\t")
                f.write("crp_prediction_time" + "\t")
                f.write("crp_transition_time" + "\n")

        processed_folders = set()
        if os.path.exists(result_txt):
            with open(result_txt, "r", encoding='ISO-8859-1') as f:
                for line in f:
                    processed_folders.add(line.split('\t')[0])

        for folder_name in tqdm(os.listdir(args.folder)):
            folder_path = os.path.join(args.folder, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            if folder_name in processed_folders:
                continue

            info_path = os.path.join(folder_path, 'info.txt')
            html_path = os.path.join(folder_path, 'html.txt')
            shot_path = os.path.join(folder_path, 'shot.png')
            predict_path = os.path.join(folder_path, 'predict.png')
            if not os.path.exists(shot_path):
                continue

            try:
                if os.path.exists(info_path) and len(open(info_path, encoding='ISO-8859-1').read().strip()) > 0:
                    url = open(info_path, encoding='ISO-8859-1').read().strip()
                else:
                    url = 'https://' + folder_name
            except FileNotFoundError:
                url = 'https://' + folder_name

            try:
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
            except (WebDriverException) as e:
                print(f"Driver crashed or encountered an error for folder {folder_name}: {e}. Restarting driver.")
                driver = restart_driver(driver)
                continue

            try:
                with open(result_txt, "a+", encoding='utf-8') as f:
                    f.write(f"{folder_name}\t{url}\t{pred}\t{F_brand}\t{F_intent}\t{brand_recog_time}\t{crp_prediction_time}\t{crp_transition_time}\n")

                if pred == 'phish':
                    plotvis.save(predict_path)
            except UnicodeEncodeError as e:
                print(f"编码错误写入文件时为文件夹 {folder_name}: {e}")
                continue

    driver.quit()
    print("\n检测完成。")