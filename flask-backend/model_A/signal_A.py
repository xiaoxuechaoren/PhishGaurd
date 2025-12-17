import argparse
import pandas as pd
import numpy as np
import pickle
import re
from urllib.parse import urlparse
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from itertools import groupby

def extract_manual_nlp_features(url):
    features = {}
    # 处理空值和非字符串类型的URL
    if not url or not isinstance(url, str):
        # 空URL返回全0特征
        features["subdomain_count"] = 0
        features["is_known_tld"] = 0
        features["has_punycode"] = 0
        features["domain_length"] = 0
        features["has_ip_address"] = 0
        features["has_at_symbol"] = 0
        features["has_double_slash"] = 0
        features["tld_in_path"] = 0
        features["domain_contains_hyphen"] = 0
        features["domain_contains_digit"] = 0
        features["word_count"] = 0
        features["avg_word_length"] = 0
        features["max_word_length"] = 0
        features["min_word_length"] = 0
        features["digit_count"] = 0
        features["special_char_count"] = 0
        features["uppercase_count"] = 0
        features["lowercase_count"] = 0
        features["consecutive_char_repeat"] = 0
        features["url_length"] = 0
        features["has_brand"] = 0
        features["has_phish_keywords"] = 0
        features["brand_in_domain"] = 0
        features["phish_keyword_in_path"] = 0
        features["long_path"] = 0
        features["has_query_parameters"] = 0
        features["query_parameter_count"] = 0
        features["suspicious_query_words"] = 0
        features["similar_brand_count"] = 0
        features["has_year_in_url"] = 0
        features["has_https"] = 0
        features["has_www"] = 0
        features["domain_www_count"] = 0
        features["path_depth"] = 0
        features["has_anchor"] = 0
        features["anchor_length"] = 0
        features["has_javascript"] = 0
        features["has_iframe"] = 0
        features["has_css"] = 0
        features["has_image"] = 0
        return features
    
    # 正常URL处理逻辑
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    url_lower = url.lower()
    
    # 域名相关特征
    features["subdomain_count"] = len(domain.split('.')) - 2 if len(domain.split('.')) >= 2 else 0
    features["is_known_tld"] = 1 if domain.endswith(('.com', '.org', '.cn', '.net', '.edu', '.gov')) else 0
    features["has_punycode"] = 1 if 'xn--' in domain else 0
    features["domain_length"] = len(domain)
    features["has_ip_address"] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0
    features["has_at_symbol"] = 1 if '@' in url else 0
    features["has_double_slash"] = 1 if '//' in path or '//' in query else 0
    features["tld_in_path"] = 1 if any(tld in path for tld in ['.com', '.org', '.cn', '.net']) else 0
    features["domain_contains_hyphen"] = 1 if '-' in domain else 0
    features["domain_contains_digit"] = 1 if any(c.isdigit() for c in domain) else 0
    
    # 文本结构特征
    url_words = re.split(r'[/-@?&=_.]', url)
    url_words = [w for w in url_words if w.strip()]
    features["word_count"] = len(url_words)
    features["avg_word_length"] = np.mean([len(w) for w in url_words]) if url_words else 0
    features["max_word_length"] = max([len(w) for w in url_words]) if url_words else 0
    features["min_word_length"] = min([len(w) for w in url_words]) if url_words else 0
    features["digit_count"] = sum(c.isdigit() for c in url)
    features["special_char_count"] = sum(1 for c in url if not c.isalnum() and c not in ':/')
    features["uppercase_count"] = sum(c.isupper() for c in url)
    features["lowercase_count"] = sum(c.islower() for c in url)
    
    # 关键修正：检查列表是否为空
    consecutive_groups = [len(list(g)) for k, g in groupby(url) if len(list(g)) >= 2]
    features["consecutive_char_repeat"] = max(consecutive_groups) if consecutive_groups else 0
    
    features["url_length"] = len(url)
    
    # 语义关联特征
    brand_words = ['paypal', 'amazon', 'alibaba', 'bank', 'wechat', 'alipay', 'facebook', 'google', 'apple', 'twitter']
    features["has_brand"] = 1 if any(brand in url_lower for brand in brand_words) else 0
    phish_keywords = ['login', 'verify', 'secure', 'update', 'password', 'account', 'signin', 'check', 'confirm', 'payment']
    features["has_phish_keywords"] = 1 if any(kw in url_lower for kw in phish_keywords) else 0
    features["brand_in_domain"] = 1 if any(brand in domain.lower() for brand in brand_words) else 0
    features["phish_keyword_in_path"] = 1 if any(kw in path.lower() for kw in phish_keywords) else 0
    features["long_path"] = 1 if len(path) > 50 else 0
    features["has_query_parameters"] = 1 if len(query) > 0 else 0
    features["query_parameter_count"] = len(query.split('&')) if query else 0
    features["suspicious_query_words"] = 1 if any(kw in query.lower() for kw in ['password', 'credit', 'card', 'ssn', 'pin']) else 0
    
    # 修正：避免brand[:-1]为空（当brand长度为1时）
    features["similar_brand_count"] = 0
    for brand in brand_words:
        if len(brand) >= 2:
            pattern1 = rf'{brand[:-1]}\d+'
            pattern2 = rf'{brand[:-2]}\d{2,}'
            if re.search(pattern1, url_lower) or re.search(pattern2, url_lower):
                features["similar_brand_count"] += 1
                
    features["has_year_in_url"] = 1 if re.search(r'20\d{2}|21\d{2}', url) else 0
    
    # 其他特征
    features["has_https"] = 1 if url.startswith('https') else 0
    features["has_www"] = 1 if 'www.' in domain else 0
    features["domain_www_count"] = domain.count('www.')
    features["path_depth"] = len(path.split('/')) - 1 if path else 0
    features["has_anchor"] = 1 if '#' in url else 0
    features["anchor_length"] = len(url.split('#')[-1]) if '#' in url else 0
    features["has_javascript"] = 1 if 'javascript:' in url_lower else 0
    features["has_iframe"] = 1 if 'iframe' in url_lower else 0
    features["has_css"] = 1 if '.css' in url_lower else 0
    features["has_image"] = 1 if any(img in url_lower for img in ['.jpg', '.png', '.gif', '.jpeg']) else 0
    
    return features

def load_detection_model(model_path="phishing_detection_model.h5", 
                         tokenizer_path="char_tokenizer.pkl", 
                         scaler_path="feature_scaler.pkl"):
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, tokenizer, scaler

def predict_phishing(url, model, tokenizer, scaler, max_seq_length=150):
    manual_features = extract_manual_nlp_features(url)
    manual_features_df = pd.DataFrame([manual_features])
    manual_features_scaled = scaler.transform(manual_features_df)
    
    char_seq = tokenizer.texts_to_sequences([url])
    char_seq_padded = pad_sequences(char_seq, maxlen=max_seq_length, padding='post', truncating='post')
    
    prob = model.predict([manual_features_scaled, char_seq_padded], verbose=0)[0][0]
    label = "钓鱼URL" if prob > 0.5 else "合法URL"
    return label, prob

def main():
    parser = argparse.ArgumentParser(description="基于URL的钓鱼网站检测工具")
    parser.add_argument("--url", required=True, help="需要检测的URL（如：https://paypa1-login-2024.com）")
    args = parser.parse_args()
    
    print("正在加载检测模型...")
    model, tokenizer, scaler = load_detection_model()
    
    label, prob = predict_phishing(args.url, model, tokenizer, scaler)
    
    print("\n检测结果：")
    print(f"URL：{args.url}")
    print(f"预测结果：{label}")
    print(f"风险概率：{prob:.4f}")

if __name__ == "__main__":
    main()